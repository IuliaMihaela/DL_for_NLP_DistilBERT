from datasets import load_dataset, concatenate_datasets, IterableDataset, interleave_datasets
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from itertools import chain

def get_paper_corpus(tokenizer_name="bert-base-uncased", streaming=False):
    """
    Paper corpus: English Wikipedia + BookCorpus (as in BERT/DistilBERT papers),
    if available via datasets.
    """
    wiki = load_dataset("wikipedia", "20220301.en", split="train", streaming=streaming, trust_remote_code=True) # alt: other snapshots
    book = load_dataset("bookcorpus", split="train", streaming=streaming, trust_remote_code=True)

    # keep only text columns
    if not streaming:
        wiki = wiki.remove_columns([c for c in wiki.column_names if c != "text"])
        book = book.remove_columns([c for c in book.column_names if c != "text"])

    if streaming:
        return interleave_datasets([wiki, book])

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
    ds = get_paper_corpus(tokenizer_name, streaming=streaming)

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=True)

    tokenized = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text", "id", "title", "url"] if not streaming else None,
        desc="Tokenizing",
    )

    # Group texts into contiguous chunks (RoBERTa-style)
    def group_texts(examples):
        # concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_len = len(concatenated["input_ids"])
        if total_len >= block_size:
            total_len = (total_len // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_len, block_size)]
            for k, t in concatenated.items()
        }
        return result

    lm_ds = tokenized.map(group_texts, batched=True, desc="Grouping into blocks")

    # For streaming datasets,  shuffle buffer
    if streaming:
        lm_ds = lm_ds.shuffle(buffer_size=10_000)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

    loader = DataLoader(
        lm_ds,
        batch_size=batch_size,
        #shuffle=True,
        shuffle=(not streaming),
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    return loader, tokenizer
