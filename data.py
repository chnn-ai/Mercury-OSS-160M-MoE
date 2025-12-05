import tiktoken
import torch
import torch.nn as nn
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset, load_from_disk

tokenizer = tiktoken.get_encoding('gpt2')


class GPTDatasetV1(Dataset):
    def __init__(self, tokenized_data, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        if isinstance(tokenized_data[0], list):
            flat_data = []
            eot_doc = tokenizer.eot_token
            for doc in tokenized_data:
                flat_data.extend(doc)
                flat_data.append(eot_doc)
            tokenized_data = flat_data

        for i in range(0, len(tokenized_data) - max_length, stride):
            input_chunks = tokenized_data[i: i + max_length]
            target_chunks = tokenized_data[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunks))
            self.target_ids.append(torch.tensor(target_chunks))

    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        #let's try with this
        if isinstance(idx, list):
            inputs = [self.input_ids[i] for i in idx]
            targets = [self.target_ids[i] for i in idx]
            return torch.stack(inputs), torch.stack(targets)
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(text, batch_size = 4, max_length = 48, stride = 128, shuffle = True,
                         drop_last = True, num_workers = 0, pin_memory = True, distributed = False):
    
    dataset = GPTDatasetV1(text, max_length, stride)
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle = shuffle)
        shuffle = False
    dataloader = DataLoader(dataset,
                            batch_size = batch_size,
                            shuffle = shuffle,
                            sampler = sampler,
                            drop_last = drop_last, 
                            num_workers = num_workers,
                            pin_memory= pin_memory)   
    return dataloader, sampler


