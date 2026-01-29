# scripts/prefetch_datasets.py
import argparse
from datasets import load_dataset

GLUE_TASKS = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, default="data/hf_cache")
    ap.add_argument("--glue", action="store_true")
    ap.add_argument("--imdb", action="store_true")
    ap.add_argument("--squad", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    cache_dir = args.cache_dir

    if args.all or args.glue:
        for task in GLUE_TASKS:
            print(f"[GLUE] downloading {task}")
            load_dataset("glue", task, cache_dir=cache_dir)

    if args.all or args.imdb:
        print("[IMDb] downloading imdb")
        load_dataset("imdb", cache_dir=cache_dir)

    if args.all or args.squad:
        print("[SQuAD v1.1] downloading squad")
        load_dataset("squad", cache_dir=cache_dir)

    print(f"Done. Cache at: {cache_dir}")

if __name__ == "__main__":
    main()
