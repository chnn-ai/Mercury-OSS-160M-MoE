import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def get_dtype():
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    

cfg = {
    "theta_base": 10.000,
    'context_length': 512,
    "vocab_size":  50257 ,
    "embed_dim": 768 ,
    "num_layers": 12 ,
    "num_heads": 12  ,
    "kv_group": 3  ,
    "num_expert":  8 ,
    "num_expert_per_tok": 1  ,
    "moe_intermediate":  469   ,
    "dtype":  get_dtype()
}
def compute_rope_params( head_dim, theta_base, context_length, dtype=torch.bfloat16, device = "cuda"):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype, device= device)[: (head_dim // 2)].float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype, device= device)

    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # Shape: (context_length, head_dim // 2)

    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    return cos, sin

def apply_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    cos = cos.to(x.device)
    sin = sin.to(x.device)

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


class GroupQueryAttention(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        
        self.d_out = cfg["embed_dim"]
        self.num_heads = cfg["num_heads"]
        self.kv_group = cfg["kv_group"]
        assert self.d_out % self.num_heads == 0, "d_out must be divisible by num_heads"
        assert self.num_heads % self.kv_group == 0, "num_heads must be divisible by kv_group"

        self.head_dim = self.d_out // self.num_heads
        self.group_size = self.num_heads // self.kv_group

        self.wq = nn.Linear(self.d_out, self.d_out, bias = False, dtype = cfg["dtype"])
        self.wk = nn.Linear(self.d_out, self.kv_group * self.head_dim, bias = False, dtype = cfg["dtype"])
        self.wv = nn.Linear(self.d_out, self.kv_group * self.head_dim, bias = False, dtype = cfg["dtype"])
        self.out_proj = nn.Linear(self.d_out, self.d_out, bias = False, dtype = cfg["dtype"])

        cos, sin = compute_rope_params(
            head_dim= self.head_dim,
            theta_base= cfg["theta_base"],
            context_length= cfg["context_length"],
            dtype= cfg['dtype'],
            device = "cuda"
        )

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)


    def forward(self, x, mask, cache = None):
        b, seq_len, dim = x.shape

        Query = self.wq(x)
        Key = self.wk(x)
        Value = self.wv(x)

        Query = Query.view(b, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        Key = Key.view(b, seq_len, self.kv_group, self.head_dim).transpose(1,2)
        Value = Value.view(b, seq_len, self.kv_group, self.head_dim).transpose(1,2)


        Query = apply_rope(Query, self.cos, self.sin)
        Key = apply_rope(Key, self.cos, self.sin)


        if cache is not None:
            
            cache_k, cache_v = Key, Value 
            cache_k = torch.cat([Key, self.cache_k], dim = 2)
            cache_v = torch.cat([Value, self.cache_v], dim = 2)
            Key, Value = cache_k,cache_v
        else:
            Key, Value = Key, Value 
            

        Key = Key.repeat_interleave(self.group_size, dim = 1)
        Value = Value.repeat_interleave(self.group_size, dim = 1)

        attention_scores = Query @ Key.transpose(2,3)
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))
        attention_weights = torch.softmax(attention_scores / Key.shape[-1] ** 0.5, dim = -1)
        attention_weights = attention_weights @ Value

        context = attention_weights.transpose(1,2).reshape(b, seq_len, self.d_out)
        return self.out_proj(context)
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = cfg["embed_dim"]
        self.intermediate_size = cfg["intermediate_size"]

        self.w1 = nn.Linear(self.embed_dim, self.intermediate_size, bias = False, dtype = cfg["dtype"])
        self.w2 = nn.Linear(self.embed_dim, self.intermediate_size, bias = False, dtype = cfg["dtype"])
        self.w3 = nn.Linear(self.intermediate_size, self.embed_dim, bias = False, dtype = cfg["dtype"])

    def forward(self,x):
        hidden_states = nn.functional.silu(self.w1(x)) * self.w2(x)
        return self.w3(hidden_states)



        
class MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = cfg["embed_dim"]
        self.num_expert_per_tok = cfg["num_expert_per_tok"]
        self.num_expert = cfg["num_expert"]

        self.gate = nn.Linear(self.embed_dim, self.num_expert, dtype = cfg["dtype"])
        self.fc1 = nn.ModuleList([nn.Linear(self.embed_dim, cfg["moe_intermediate"], bias = False, dtype = cfg["dtype"])
                                  for _ in range(self.num_expert)])
        self.fc2 = nn.ModuleList([nn.Linear(self.embed_dim, cfg["moe_intermediate"], bias = False, dtype = cfg["dtype"])
                                  for _ in range(self.num_expert)])
        self.fc3 = nn.ModuleList([nn.Linear(cfg["moe_intermediate"], self.embed_dim, bias = False, dtype = cfg["dtype"])
                                  for _ in range(self.num_expert)])
        
    def forward(self,x):
        gate = self.gate(x)
        values, indices = torch.topk(gate, self.num_expert_per_tok, dim = -1)
        probs = torch.softmax(values, dim = -1)

        b, seq_len, _ = x.shape
        x_flat = x.reshape(b * seq_len, -1)
        out_flat = torch.zeros(b * seq_len, self.embed_dim, device = x.device, dtype = x.dtype)

        indices_f = indices.reshape(-1, self.num_expert_per_tok)
        probs_f = probs.reshape(-1, self.num_expert_per_tok)

        unique_experts = torch.unique(indices_f)

        for expert_tensor in unique_experts:
            expert_id = int(expert_tensor.item())
            mask = indices_f == expert_id

            token_mask = mask.any(dim = -1)
            selected_idx = token_mask.nonzero().squeeze(-1)
            if selected_idx.numel() == 0:
                continue

            expert_input = x_flat.index_select(0, selected_idx)
            hidden = nn.functional.silu(self.fc1[expert_id](expert_input)) * self.fc2[expert_id](expert_input)
            out = self.fc3[expert_id](hidden)

            mask_selec = mask[selected_idx]
            slot_indices = mask_selec.int().argmax(dim = -1, keepdim = True)
            select_probs = torch.gather(probs_f.index_select(0, selected_idx), dim = -1, index = slot_indices).squeeze(-1)
            out_flat.index_add_(0,selected_idx, out * select_probs.unsqueeze(-1))

        return out_flat.reshape(b, seq_len, self.embed_dim)
    
class RMS_Norm(nn.Module):
    def __init__(self, cfg, eps = 1e-6):
        super().__init__()
        self.embed_dim = cfg["embed_dim"]
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(self.embed_dim))

    def forward(self,x):
        RMS = torch.sqrt(torch.mean(x ** 2, dim = -1, keepdim = True) + self.eps)
        norm = x/RMS
        return self.gamma * norm
    


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.rmsnorm = RMS_Norm(cfg)
        self.gqa = GroupQueryAttention(cfg)
        self.moe = MoE(cfg)

    def forward(self,x, mask):
        shortcut = x
        x = self.rmsnorm(x)
        x = self.gqa(x, mask)
        x = x + shortcut
        
        shortcut = x
        x = self.rmsnorm(x)
        x = self.moe(x)
        x = shortcut + x

        return x

#add checkpoints to save memory
def checkpointed_block(block, x, mask):
    return block(x, mask)


class MercuryOSS(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.current_pos = 0
        self.embedding = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"], dtype = cfg["dtype"])
        self.blocks = nn.ModuleList([Transformer(cfg) for _ in range(cfg["num_layers"])])

        self.fn = RMS_Norm(cfg)
        #we use tie embedding so the final layer is tied with self.embedding
        #self.fl = self.embedding.weight.T is not defined here because it gets stored in the cpu


    def forward(self,idx, cache = None):
        b, seq_len = idx.shape
        x = self.embedding(idx)

        num_tokens = x.shape[1]
        if cache is not None:
            pos_start = self.current_pos
            pos_end = pos_start + num_tokens
            self.current_pos = pos_end
            mask = torch.triu(
                torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool), diagonal=1
            )[pos_start:pos_end, :pos_end]
        else:
            pos_start = 0 
            mask = torch.triu(
                torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1
            )
        # Shape (1, 1, num_tokens, num_tokens) to broadcast across batch and heads
        mask = mask[None, None, :, :].bool()

        for block in self.blocks:
            if self.training:
                x = checkpoint(checkpointed_block(block, x, mask), block, x, mask, use_reentrant=False)
            else:
                x = block(x, mask)
        
        x = self.fn(x)
        x = x @ self.embedding.weight.T
        return x 

  

