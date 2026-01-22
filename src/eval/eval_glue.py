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
    set_seed,
)

from utils import set_all_seeds, median, save_json, print_table

GLUE_TASKS = ["cola","mnli","mrpc","qnli","qqp","rte","sst2","stsb","wnli"]

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def metric_key(task: str):
    if task == "stsb":
        return "pearson"
    if task == "cola":
        return "matthews_correlation"
    return "accuracy"

def is_regression(task: str) -> bool:
    return task == "stsb"

def preprocess_fn(tokenizer, task, max_length: int):
    k1, k2 = TASK_TO_KEYS[task]
    def fn(ex):
        a = ex[k1]
        b = ex[k2] if k2 is not None else None
        return tokenizer(a, b, truncation=True, max_length=max_length)
    return fn

def run_task(model_name_or_path: str, task: str, seed: int, out_dir: str, max_length: int, bs: int, lr: float, epochs: float):
    set_all_seeds(seed)
    set_seed(seed)

    ds = load_dataset("glue", task)
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    reg = is_regression(task)
    num_labels = 1 if reg else ds["train"].features["label"].num_classes

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        problem_type="regression" if reg else None
    )

    ds_tok = ds.map(preprocess_fn(tok, task, max_length), batched=True)
    keep = {"input_ids","attention_mask","label"}
    drop_cols = [c for c in ds_tok["train"].column_names if c not in keep]
    ds_tok = ds_tok.remove_columns(drop_cols)

    metric = evaluate.load("glue", task)

    def compute_metrics(p):
        preds = p.predictions
        labels = p.label_ids
        if reg:
            preds = np.squeeze(preds)
        else:
            preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=labels)

    eval_split = "validation_matched" if task == "mnli" else "validation"

    args = TrainingArguments(
        output_dir=os.path.join(out_dir, f"{task}_seed{seed}"),
        learning_rate=lr,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_strategy="no",
        report_to=[],
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok[eval_split],
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    m = trainer.evaluate()

    key = metric_key(task)
    val = m.get(f"eval_{key}", m.get(key))
    return float(val)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="bert-base-uncased OR checkpoints/final")
    ap.add_argument("--out_dir", type=str, default="eval_out/glue")
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=3.0)
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    macro_vals = []

    for task in GLUE_TASKS:
        vals = []
        for s in seeds:
            v = run_task(args.model, task, s, args.out_dir, args.max_length, args.batch_size, args.lr, args.epochs)
            vals.append(v)
            print(f"{task} seed={s}: {v:.4f}")
        med = median(vals)
        results[task] = {"runs": vals, "median": med}
        macro_vals.append(med)

    macro = float(np.mean(macro_vals))

    save_json(
        {"model": args.model, "seeds": seeds, "tasks": results, "macro_score": macro},
        os.path.join(args.out_dir, "table1_glue.json")
    )

    rows = []
    for t in GLUE_TASKS:
        rows.append([t, f"{results[t]['median']:.4f}", str([round(x,4) for x in results[t]["runs"]])])

    print_table("TABLE 1 (GLUE dev) - medians of runs", ["task", "median", "runs"], rows)
    print("\nMacro-score (avg of task medians):", round(macro, 4))

if __name__ == "__main__":
    main()
