import torch
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, default_data_collator
from datasets import load_dataset

def get_paper_corpus(tokenizer_name, streaming=True):
    """
    Loads WikiText-103, which is a large, clean subset of Verified Wikipedia articles.
    This replaces the old 'wikipedia' dataset which is no longer supported by modern
    Hugging Face versions due to security changes.
    """
    # Load WikiText-103 (Raw version, train split)
    #ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=streaming, revision="main")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=streaming)
    
    # WikiText has a 'text' column, but some rows are empty headers. We filter them.
    ds = ds.filter(lambda x: len(x["text"]) > 10)
    
    return ds

def build_mlm_dataloader(tokenizer_name, block_size=128, batch_size=8, mlm_probability=0.15, streaming=True):
    """
    Builds a streaming DataLoader for Masked Language Modeling (MLM).
    """
    dataset = get_paper_corpus(tokenizer_name, streaming=streaming)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 1. Tokenize function
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    # 2. Apply tokenization
    # Note: When streaming, we use map() directly. 
    # We allow the tokenizer to run on variable lengths, then we group them below.
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 3. Grouping function (concatenates texts and chops them into block_size chunks)
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
            
        # Split by chunks of max_len
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    # 4. Apply grouping
    # Using a large batch size for mapping ensures we have enough tokens to fill blocks
    lm_datasets = tokenized_datasets.map(group_texts, batched=True, batch_size=1000)

    # 5. Data Collator (Handles the [MASK]ing automatically)
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
    )

    # 6. Build Loader
    loader = DataLoader(
        lm_datasets,
        batch_size=batch_size,
        collate_fn=data_collator
    )
    
    return loader, tokenizer