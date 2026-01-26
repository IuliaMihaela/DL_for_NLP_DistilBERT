from datasets import load_dataset
from transformers import BertTokenizer, DataCollatorForLanguageModeling
import torch

class DistillationDataset:
    def __init__(self, model_name="bert-base-uncased", subset_size=10000):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.dataset = load_dataset('text', data_files={'train': 'data/train.txt'})['train']
        self.dataset = self.dataset.select(range(min(subset_size, len(self.dataset))))

    def preprocess_function(self, examples):
        return self.tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=256 # Increased for better attention resolution
        )

    def get_data_loader(self, batch_size=8):
        tokenized = self.dataset.map(self.preprocess_function, batched=True, remove_columns=["text"])
        tokenized.set_format("torch")
        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)
        return torch.utils.data.DataLoader(tokenized, batch_size=batch_size, collate_fn=collator)