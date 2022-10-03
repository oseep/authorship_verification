import os
import sys
sys.path.append("../")

import pickle
import json
import glob
from tqdm.auto import trange, tqdm
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel
from features import merge_entries, prepare_entry
import nltk
from utills import chunker
import torch
from torch.utils.data import Dataset, random_split
import gc
import wandb
import numpy as np


PREPROCESSED_DATA_PATH = '/scratch/jnw301/av_public/temp_data/pan/'
TEMP_DATA_PATH = '/scratch/jnw301/av_public/temp_data/pan/finetuning/'

class PANDataset(Dataset):
    def __init__(self, input_ids_path, attention_mask_path, num_records, max_length, max_records=None):
        
        self.input_ids = np.memmap(input_ids_path, dtype='int32', mode='r', shape=(num_records, max_length))
        self.attention_mask = np.memmap(attention_mask_path, dtype='int32', mode='r', shape=(num_records, max_length))
        self.num_records = num_records
        if max_records is not None:
            self.idxs = np.random.choice(num_records, max_records)
        else:
            self.idxs = np.arange(num_records)
            
    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        idx = self.idxs[idx]
        return torch.LongTensor(np.array(self.input_ids[idx])), torch.LongTensor(np.array(self.attention_mask[idx]))

    
if __name__ == "__main__":
    wandb.login()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium', bos_token='<|startoftext|>',
                                              eos_token='<|endoftext|>', pad_token='<|pad|>')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium').to(device)
    model.resize_token_embeddings(len(tokenizer))

    with open(TEMP_DATA_PATH + 'metadata.p', 'rb') as f:
        num_records, max_length = pickle.load(f)

    dataset = PANDataset(TEMP_DATA_PATH + 'input_ids.npy', TEMP_DATA_PATH + 'attention_mask.npy', num_records, max_length, max_records=10000)
    train_size = int(0.9 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    gc.collect()
    torch.cuda.empty_cache()
    training_args = TrainingArguments(
        output_dir=TEMP_DATA_PATH + 'results_finetune', 
        num_train_epochs=10, 
        logging_steps=50, 
        save_steps=5000, 
        learning_rate=0.01,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=1,
        #evaluate_during_training=True,
        evaluation_strategy="steps",
        warmup_steps=10,
        logging_dir='./logs',
        report_to = 'wandb'
    )

    Trainer(model=model,  args=training_args, train_dataset=train_dataset, 
            eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                                  'attention_mask': torch.stack([f[1] for f in data]),
                                                                  'labels': torch.stack([f[0] for f in data])}).train()
    torch.save(model, TEMP_DATA_PATH + 'gpt2_medium_fanfictions.pt')
    wandb.finish()