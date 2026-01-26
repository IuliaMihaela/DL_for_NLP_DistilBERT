# dataset.py

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset

# --- CRITICAL IMPORT ---
# This connects your training loop to the massive streaming data pipeline
from paper_corpus import build_mlm_dataloader 
# -----------------------

class DistillationDataset:
    def __init__(
        self,
        model_name: str,
        subset_size: int = 10000,
        mode: str = "small",          # "small" | "paper"
        data_dir: str | None = None,  # for paper mode caching/loading
        max_length: int = 128,
        mlm_probability: float = 0.15,
    ):
        self.mode = mode
        self.subset_size = subset_size
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.model_name = model_name
        self.data_dir = data_dir

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # We only preload data here if we are in "small" mode.
        # If we are in "paper" mode, we delay loading until get_data_loader() is called
        # to properly handle streaming.
        if self.mode == "small":
            self.dataset = self._load_small_dataset()
            # Set format for the manual masking logic below
            self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    def get_data_loader(self, batch_size: int):
        """
        Returns the appropriate DataLoader based on the mode.
        """
        # --------------------------------------------------
        # MODE: PAPER (Delegates to paper_corpus.py)
        # --------------------------------------------------
        if self.mode == "paper":
            # This handles downloading Wikipedia, Streaming, and Collating automatically
            loader, _ = build_mlm_dataloader(
                tokenizer_name=self.model_name,
                block_size=self.max_length,
                batch_size=batch_size,
                mlm_probability=self.mlm_probability,
                streaming=True  # Ensure we don't crash RAM
            )
            return loader

        # --------------------------------------------------
        # MODE: SMALL (Local Debugging)
        # --------------------------------------------------
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
        )

    def _load_small_dataset(self):
        # Load WikiText-2 (small)
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

        if self.subset_size:
            ds = ds.select(range(min(self.subset_size, len(ds))))

        # Apply tokenization and masking immediately for small datasets
        ds = ds.map(
            self._tokenize_and_mask,
            batched=True,
            remove_columns=ds.column_names,
        )
        return ds

    def _tokenize_and_mask(self, examples):
        """
        Manual MLM implementation for the small dataset.
        """
        enc = self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

        input_ids = torch.tensor(enc["input_ids"])
        labels = input_ids.clone()

        # Create masking matrix
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # Mask special tokens (CLS, SEP, PAD) so we don't try to predict them
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), 0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # -100 means "ignore this in loss calculation"

        # 80% of time, replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of time, replace with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        return {
            "input_ids": input_ids,
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "labels": labels,
        }
