import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset


class DistillationDataset:
    def __init__(
        self,
        model_name: str,
        subset_size: int = 10000,
        mode: str = "small",          # "small" | "paper"
        data_dir: str | None = None,  # для paper-режима
        max_length: int = 128,
        mlm_probability: float = 0.15,
    ):
        self.mode = mode
        self.subset_size = subset_size
        self.max_length = max_length
        self.mlm_probability = mlm_probability

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.mode == "small":
            self.dataset = self._load_small_dataset()
        elif self.mode == "paper":
            self.dataset = self._load_paper_dataset(data_dir)
        else:
            raise ValueError(f"Unknown dataset mode: {mode}")
        
        self.dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )


    # --------------------------------------------------
    # SMALL DATASET (debug / sanity / CI)
    # --------------------------------------------------
    def _load_small_dataset(self):
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

        if self.subset_size:
            ds = ds.select(range(min(self.subset_size, len(ds))))

        ds = ds.map(
            self._tokenize_and_mask,
            batched=True,
            remove_columns=ds.column_names,
        )
        return ds

    # --------------------------------------------------
    # PAPER DATASET (Wikipedia + BookCorpus)
    # --------------------------------------------------
    def _load_paper_dataset(self, data_dir):
        if data_dir is None:
            raise ValueError("paper dataset requires data_dir")

        # предполагаем, что ты подготовишь файлы заранее
        ds = load_dataset(
            "text",
            data_files={
                "train": f"{data_dir}/train.txt",
            },
            split="train",
        )

        if self.subset_size:
            ds = ds.select(range(min(self.subset_size, len(ds))))

        ds = ds.map(
            self._tokenize_and_mask,
            batched=True,
            remove_columns=ds.column_names,
        )
        return ds

    # --------------------------------------------------
    # TOKENIZATION + MLM MASKING
    # --------------------------------------------------
    def _tokenize_and_mask(self, examples):
        enc = self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

        input_ids = torch.tensor(enc["input_ids"])
        labels = input_ids.clone()

        # MLM masking
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), 0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 80% [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% random
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        return {
            "input_ids": input_ids,
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "labels": labels,
        }

    # --------------------------------------------------
    def get_data_loader(self, batch_size: int):
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
        )
