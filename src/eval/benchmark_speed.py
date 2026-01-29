import os
import time
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--max_samples", type=int, default=None,
                    help="limit number of STS-B validation samples (e.g. 200, 500, 1000)")

    ap.add_argument("--cache_dir", type=str, default="data/hf_cache")
    ap.add_argument("--offline", action="store_true", help="use only local HF cache (no downloads)")
    args = ap.parse_args()

    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    os.makedirs(args.cache_dir, exist_ok=True)

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    # Load STS-B validation set
    ds = load_dataset("glue", "stsb", cache_dir=args.cache_dir)["validation"]

    # Limit samples if requested
    if args.max_samples is not None:
        n = min(int(args.max_samples), len(ds))
        ds = ds.select(range(n))

    tok = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, cache_dir=args.cache_dir, local_files_only=args.offline
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, cache_dir=args.cache_dir, local_files_only=args.offline
    ).to(device)
    model.eval()

    n = len(ds)
    s1 = [str(x) for x in ds["sentence1"][:n]]
    s2 = [str(x) for x in ds["sentence2"][:n]]

    # Tokenize once (so timing measures mostly model forward)
    enc = tok(s1, s2, truncation=True, max_length=args.max_length)

    # Warmup (always uses first example)
    for _ in range(args.warmup):
        input_ids = torch.tensor([enc["input_ids"][0]], dtype=torch.long)
        attn = torch.tensor([enc["attention_mask"][0]], dtype=torch.long)
        _ = model(input_ids=input_ids, attention_mask=attn)

    # Timing: full pass over N samples
    t0 = time.perf_counter()
    for i in range(n):
        input_ids = torch.tensor([enc["input_ids"][i]], dtype=torch.long)
        attn = torch.tensor([enc["attention_mask"][i]], dtype=torch.long)
        _ = model(input_ids=input_ids, attention_mask=attn)
    t1 = time.perf_counter()

    secs = t1 - t0
    print(f"Model: {args.model}")
    print(f"# param (Millions): {round(count_params_m(model), 2)}")
    print(f"Samples: {n}")
    print(f"Inf. time STS-B pass (CPU, batch=1): {round(secs, 2)} seconds")
    if n > 0:
        print(f"Throughput: {round(n / secs, 2)} samples/sec")

if __name__ == "__main__":
    main()
