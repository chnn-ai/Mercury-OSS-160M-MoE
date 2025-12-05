import os
import torch
import torch.nn as nn
import tiktoken
import argparse
import pickle
import wandb
import torch.distributed as dist 
from model import cfg, MercuryOSS   
from data import create_dataloader_v1 
from torch.nn.parallel import DistributedDataParallel as DDP
from huggingface_hub import hf_hub_download 
from transformers import get_cosine_schedule_with_warmup
from muon import get_optimizer


def setup_ddp():
    # Initialize the distributed process group.
    dist.init_process_group(backend="nccl")
    # Get local rank from environment variable
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

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



def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):  
        logits = model(input_batch)
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
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

    total_loss_tensor = torch.tensor(total_loss / num_batches, device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.AVG)       

    return total_loss_tensor.item()



def generate_and_print_sample(model, tokenizer, device, start_context):

    if int(os.environ["RANK"]) != 0:
        return
    model.eval()
    raw_model = model.module if hasattr(model, "module") else model
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
def train_model_simple(model,
                       train_loader, 
                       val_loader,
                       train_sampler,
                       optimizer,
                       scheduler,
                       device, 
                       num_epochs,
                       eval_freq,
                       eval_iter, 
                       start_context, 
                       tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    rank = int(os.environ["RANK"])
    if rank == 0:
        wandb.watch(model, log="all", log_freq=100)
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            if rank == 0 and global_step % 10 == 0:
                wandb.log({
                    "train_loss_step": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "global_step": global_step,
                    "tokens_seen_global": tokens_seen * int(os.environ["WORLD_SIZE"])
                })
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                if rank == 0:
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen * int(os.environ["WORLD_SIZE"]))
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                      )
                wandb.log({
                        "val_loss": val_loss,
                        "train_loss_epoch": train_loss,
                    })
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--optimizer", type = str, default = "muon")
    parser.add_argument("--data_path", type=str, default="/data/fineweb_350_tokenized.pkl", help="Path to tokenized data pickle")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size (tune for H100: 256+)")
    parser.add_argument("--max_length", type=int, default=512, help="Sequence length")
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--num_workers", type = int, default = 4)
    args = parser.parse_args()

    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    global_rank = int(os.environ["RANK"])
    data_path = "/data/fineweb_350k_tokenized.pkl"

    if global_rank == 0:

        if not os.path.exists(data_path):
            print("Dataset not found locally, downloading from Hugging Face...")
            
            hf_hub_download(
            repo_id="thesouth/fineweb_350_tokenized.pkl",
            filename="fineweb_350_tokenized.pkl",
            repo_type="dataset",
            local_dir="/data",
            local_dir_use_symlinks=False
            )
            print("Download complete!")
        else:
            print("Dataset found locally, skipping download.")
            # NEW: Load data from mounted volume
            print("Loading data from", args.data_path)
    # Make other ranks wait until Rank 0 is done
    dist.barrier()

    if global_rank == 0:
        wandb.init(
            project="mercury-8xH100", 
            name=f"run-muon-bs{args.batch_size}-8gpu",
            config={
                "architecture": "MercuryOSS",
                "dataset": "FineWeb-350",
                "optimizer": args.optimizer,
                "batch_size_per_gpu": args.batch_size,
                "global_batch_size": args.batch_size * 8,
                "learning_rate": 0.008,
                "num_epochs": args.epochs,
            }
        )   
    with open(args.data_path, "rb") as f:
        tokenized_data_f = pickle.load(f)
    print(type(tokenized_data_f))

    
    
    tokenizer = tiktoken.get_encoding("gpt2")
    train_split = 0.90
    train_length = int(train_split * len(tokenized_data_f))
    train = tokenized_data_f[:train_length]
    valid = tokenized_data_f[train_length:]

    
    train_loader, train_sampler = create_dataloader_v1(
        text=train,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.stride,
        shuffle=True,
        drop_last=True,
        num_workers= args.num_workers,
        pin_memory= True,
        distributed = True
    )

    valid_loader, valid_sampler = create_dataloader_v1(
        text=valid,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.stride,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory= True,
        distributed = True
    )

    
    device = torch.device("cuda")
    model = MercuryOSS(cfg)  
    model.to(device, dtype=torch.bfloat16)  
    model = torch.compile(model)
    
    model = DDP(model, device_ids=[local_rank])
    optimizer = get_optimizer(args.optimizer, model, lr = 0.008)
    
    
    total_steps = len(train_loader) * args.epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = int(total_steps * 0.1),
        num_training_steps = total_steps

    )
    # Optional H100 optimization
    torch.backends.cudnn.benchmark = True

    if global_rank == 0:
        print(f"Starting training on {os.environ['WORLD_SIZE']} GPUs...")
        print(f"Total Steps: {total_steps}, Warmup Steps: {int(total_steps * 0.1)}")
        
        #just a little check
        for name, p in model.named_parameters():
            print(name)
            
        

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, valid_loader, train_sampler, optimizer, scheduler, device,
        num_epochs=args.epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    if global_rank == 0:
        save_path = "/data/checkpoints"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.module.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        wandb.finish()
    
    cleanup_ddp()