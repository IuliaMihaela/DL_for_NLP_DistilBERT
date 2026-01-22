from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

def get_paper_corpus(tokenizer_name="bert-base-uncased"):
    """
    Paper corpus: English Wikipedia + BookCorpus (as in BERT/DistilBERT papers),
    if available via datasets.
    """
    wiki = load_dataset("wikipedia", "20220301.en", split="train")  # alt: other snapshots
    book = load_dataset("bookcorpus", split="train")

    # keep only text columns
    wiki = wiki.remove_columns([c for c in wiki.column_names if c != "text"])
    book = book.remove_columns([c for c in book.column_names if c != "text"])

    return concatenate_datasets([wiki, book])

def build_mlm_dataloader(
    tokenizer_name="bert-base-uncased",
    block_size=512,
    batch_size=8,
    num_workers=0,
    mlm_probability=0.15,
    streaming=False,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    ds = get_paper_corpus(tokenizer_name)

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    tokenized = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    # Group texts into contiguous chunks (RoBERTa-style)
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = len(concatenated["input_ids"])
        total_len = (total_len // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_len, block_size)]
            for k, t in concatenated.items()
        }
        return result

    lm_ds = tokenized.map(group_texts, batched=True, desc="Grouping into blocks")

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

    loader = DataLoader(
        lm_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    return loader, tokenizer
