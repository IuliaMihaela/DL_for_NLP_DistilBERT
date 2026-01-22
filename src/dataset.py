import torch
from datasets import load_dataset
from transformers import BertTokenizer, DataCollatorForLanguageModeling

class DistillationDataset:
    def __init__(self, model_name="bert-base-uncased", subset_size=10000):
        # Use the standard BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
                
        #self.dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        self.dataset = load_dataset('text', data_files={'train': 'data/train.txt'})['train']
        
        # Limit size for quick testing
        self.dataset = self.dataset.select(range(min(subset_size, len(self.dataset))))
        
        if mode == "paper":
            if data_dir is None:
                raise ValueError("mode='paper' requires --data_dir with prepared corpus files")
        


    def preprocess_function(self, examples):
        # Remove empty lines and tokenize
        texts = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        return self.tokenizer(texts, truncation=True, padding="max_length", max_length=128)

    def get_data_loader(self, batch_size=8):
        # Tokenize the dataset
        tokenized_dataset = self.dataset.map(
            self.preprocess_function, 
            batched=True, 
            remove_columns=["text"]
        )
        tokenized_dataset.set_format("torch")

        # Dynamic Masking 
        # masking 15% of tokens randomly during each epoch
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=True, 
            mlm_probability=0.15
        )

        return torch.utils.data.DataLoader(
            tokenized_dataset, 
            batch_size=batch_size, 
            collate_fn=data_collator
        )