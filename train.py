import torch
import torch.nn as nn
import tiktoken
import argparse
import pickle 
from model import cfg  
from model import MercuryOSS 
from data import create_dataloader_v1 
import os
from huggingface_hub import hf_hub_download 




def tok_to_idx(text, tokenizer):
    encode = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    return encode


def idx_to_tok(output, tokenizer):
    decode = tokenizer.decode(output.squeeze(0).tolist())
    return decode



def generate_text(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        '''Crops current context if it exceeds the supported context size...'''
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        '''focuses only on the last time step...'''
        probs = torch.softmax(logits, dim=-1, )
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


print('already at line 80, calc_loss_batch')
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):  
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches






def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = cfg["context_length"]  
    encoded = tok_to_idx(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = idx_to_tok(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss


# CODE TO TRAIN OUR MODEL
def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            #check if a param or buffer is on cpu instead of gpu
            for name, p in model.named_parameters():
                if not p.is_cuda:
                    print("PARAM ON CPU:", name, p.device)
            for name, b in model.named_buffers():
                if not b.is_cuda:
                    print("BUFFER ON CPU:", name, b.device)

            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                      )
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--data_path", type=str, default="/teamspace/studios/this_studio/fineweb_350k_tokenized.pkl", help="Path to tokenized data pickle")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (tune for H100: 256+)")
    parser.add_argument("--max_length", type=int, default=512, help="Sequence length")
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--num_workers", type = int, default = 4)
    args = parser.parse_args()

    data_path = "/teamspace/studios/this_studio/fineweb_350k_tokenized.pkl"

    if not os.path.exists(data_path):
        print("Dataset not found locally, downloading from Hugging Face...")
        os.makedirs("/teamspace/studios/this_studio", exist_ok=True)
        hf_hub_download(
        repo_id="thesouth/my-fineweb-tokenized",
        filename="fineweb_350k_tokenized.pkl",
        repo_type="dataset",
        local_dir="/teamspace/studios/this_studio",
        local_dir_use_symlinks=False
        )
        print("Download complete!")
    else:
        print("Dataset found locally, skipping download.")
        # NEW: Load data from mounted volume
        print("Loading data from", args.data_path)


        
    with open(args.data_path, "rb") as f:
        tokenized_data_f = pickle.load(f)
    print(type(tokenized_data_f))

    
    
    tokenizer = tiktoken.get_encoding("gpt2")
    train_split = 0.90
    train_length = int(train_split * len(tokenized_data_f))
    train = tokenized_data_f[:train_length]
    valid = tokenized_data_f[train_length:]

    
    train_loader = create_dataloader_v1(
        text=train,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.stride,
        shuffle=True,
        drop_last=True,
        num_workers= args.num_workers,
        pin_memory= True
    )

    valid_loader = create_dataloader_v1(
        text=valid,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.stride,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory= True
    )

    
    device = torch.device("cuda")
    model = MercuryOSS(cfg)  
    model.to(device, dtype=torch.bfloat16)  
    model = torch.compile(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    # Optional H100 optimization
    torch.backends.cudnn.benchmark = True

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, valid_loader, optimizer, device,
        num_epochs=args.epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    # Save to /checkpoints
    torch.save(model.state_dict(), "/teamspace/studios/this_studio/checkpoints/gpt_final.pth")
    print(f"Losses: {train_losses}")
    print('training is starting...') 
