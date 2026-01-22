import os
import argparse
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed
)
from utils import set_all_seeds, save_json, print_table, median

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="eval_out/imdb")
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset("imdb")
    metric = evaluate.load("accuracy")

    def tokenize(tok, ex):
        return tok(ex["text"], truncation=True, max_length=args.max_length)

    runs = []
    for seed in seeds:
        set_all_seeds(seed); set_seed(seed)

        tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        ds_tok = ds.map(lambda ex: tokenize(tok, ex), batched=True)
        ds_tok = ds_tok.remove_columns([c for c in ds_tok["train"].column_names if c not in ["input_ids","attention_mask","label"]])

        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

        def compute_metrics(p):
            preds = np.argmax(p.predictions, axis=1)
            return metric.compute(predictions=preds, references=p.label_ids)

        train_args = TrainingArguments(
            output_dir=os.path.join(args.out_dir, f"seed{seed}"),
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            evaluation_strategy="epoch",
            save_strategy="no",
            logging_strategy="no",
            report_to=[],
            seed=seed,
        )

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=ds_tok["train"],
            eval_dataset=ds_tok["test"],
            tokenizer=tok,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        m = trainer.evaluate()
        acc = float(m["eval_accuracy"])
        runs.append(acc)
        print(f"IMDb seed={seed}: {acc:.4f}")

    med = median(runs)
    save_json({"model": args.model, "seeds": seeds, "runs": runs, "median": med}, os.path.join(args.out_dir, "imdb.json"))

    print_table("TABLE 2 (IMDb test accuracy)", ["model","median_acc","runs"], [[args.model, f"{med:.4f}", str([round(x,4) for x in runs])]])

if __name__ == "__main__":
    main()
