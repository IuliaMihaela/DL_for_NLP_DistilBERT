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
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    ds = load_dataset("glue", "stsb")["validation"]
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(device)
    model.eval()

    enc = tok(ds["sentence1"], ds["sentence2"], truncation=True, max_length=args.max_length)

    # warmup
    for _ in range(args.warmup):
        i = 0
        input_ids = torch.tensor([enc["input_ids"][i]], dtype=torch.long)
        attn = torch.tensor([enc["attention_mask"][i]], dtype=torch.long)
        _ = model(input_ids=input_ids, attention_mask=attn)

    t0 = time.perf_counter()
    for i in range(len(ds)):
        input_ids = torch.tensor([enc["input_ids"][i]], dtype=torch.long)
        attn = torch.tensor([enc["attention_mask"][i]], dtype=torch.long)
        _ = model(input_ids=input_ids, attention_mask=attn)
    t1 = time.perf_counter()

    print("Model:", args.model)
    print("# param (Millions):", round(count_params_m(model), 2))
    print("Inf. time STS-B full pass (CPU, batch=1):", round(t1 - t0, 2), "seconds")

if __name__ == "__main__":
    main()
