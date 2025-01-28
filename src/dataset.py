# src/dataset.py

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class TextToKGDataset(Dataset):
    def __init__(self, samples, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]
        input_text = data["input_text"]
        target_text = data["target_text"]

        # Tokenize
        tokenized_input = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )

        tokenized_target = self.tokenizer(
            target_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )

        # Convert to PyTorch tensors
        input_ids = torch.tensor(tokenized_input["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(tokenized_input["attention_mask"], dtype=torch.long)

        labels = torch.tensor(tokenized_target["input_ids"], dtype=torch.long)
        
        # We can also mask out padding tokens for the labels
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
