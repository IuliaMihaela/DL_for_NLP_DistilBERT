# src/eval/eval_squad.py
import os
import argparse
import collections
import numpy as np
import torch
import torch.nn.functional as F
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    set_seed,
)
from utils import set_all_seeds, save_json, print_table, median


def _limit_split(ds_split, max_samples: int | None, seed: int):
    """Shuffle + take first N (stable per seed)."""
    if max_samples is None:
        return ds_split
    n = min(int(max_samples), len(ds_split))
    return ds_split.shuffle(seed=seed).select(range(n))


def prepare_train_features(examples, tokenizer, max_length=384, doc_stride=128):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized["offset_mapping"]

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1

            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                start_positions.append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                end_positions.append(token_end_index + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


def prepare_validation_features(examples, tokenizer, max_length=384, doc_stride=128):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    tokenized["example_id"] = []

    for i in range(len(tokenized["input_ids"])):
        sequence_ids = tokenized.sequence_ids(i)
        context_index = 1
        sample_index = sample_mapping[i]
        tokenized["example_id"].append(examples["id"][sample_index])

        tokenized["offset_mapping"][i] = [
            o if sequence_ids[k] == context_index else None
            for k, o in enumerate(tokenized["offset_mapping"][i])
        ]
    return tokenized


def postprocess_qa_predictions(tokenizer, examples, features, raw_predictions, n_best_size=20, max_answer_length=30):
    all_start_logits, all_end_logits = raw_predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, f in enumerate(features):
        features_per_example[example_id_to_index[f["example_id"]]].append(i)

    predictions = collections.OrderedDict()

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        valid_answers = []

        context = example["context"]

        for fi in feature_indices:
            start_logits = all_start_logits[fi]
            end_logits = all_end_logits[fi]
            offset_mapping = features[fi]["offset_mapping"]

            start_indexes = np.argsort(start_logits)[-n_best_size:][::-1]
            end_indexes = np.argsort(end_logits)[-n_best_size:][::-1]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offset_mapping) or end_index >= len(offset_mapping):
                        continue
                    if offset_mapping[start_index] is None or offset_mapping[end_index] is None:
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {"score": float(start_logits[start_index] + end_logits[end_index]),
                         "text": context[start_char:end_char]}
                    )

        if len(valid_answers) > 0:
            best = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            predictions[example["id"]] = best["text"]
        else:
            predictions[example["id"]] = ""

    return predictions


class DistillQATrainer(Trainer):
    def __init__(self, teacher, temperature=2.0, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.temperature = float(temperature)
        self.beta = float(beta)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        ce_loss = outputs.loss

        with torch.no_grad():
            t_out = self.teacher(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                token_type_ids=inputs.get("token_type_ids", None),
                return_dict=True
            )

        T = self.temperature
        s_start = outputs.start_logits / T
        t_start = t_out.start_logits / T
        s_end = outputs.end_logits / T
        t_end = t_out.end_logits / T

        kl_start = F.kl_div(
            F.log_softmax(s_start, dim=-1),
            F.softmax(t_start, dim=-1),
            reduction="batchmean"
        ) * (T * T)

        kl_end = F.kl_div(
            F.log_softmax(s_end, dim=-1),
            F.softmax(t_end, dim=-1),
            reduction="batchmean"
        ) * (T * T)

        distill = 0.5 * (kl_start + kl_end)
        loss = ce_loss + self.beta * distill

        return (loss, outputs) if return_outputs else loss


def run_one(
    model_name: str,
    seed: int,
    out_dir: str,
    use_distill: bool,
    teacher_name: str | None = None,
    temperature: float = 2.0,
    beta: float = 1.0,
    cache_dir: str | None = None,
    offline: bool = False,
    # limits
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    # tokenization params
    max_length: int = 384,
    doc_stride: int = 128,
    # training params
    lr: float = 3e-5,
    train_bs: int = 12,
    eval_bs: int = 12,
    epochs: float = 2.0,
):
    set_all_seeds(seed)
    set_seed(seed)

    if offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    ds = load_dataset("squad", cache_dir=cache_dir)

    # Limit BEFORE mapping (this is where speed/memory is won)
    train_ds = _limit_split(ds["train"], max_train_samples, seed)
    eval_ds = _limit_split(ds["validation"], max_eval_samples, seed)

    tok = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, cache_dir=cache_dir, local_files_only=offline
    )

    train_feats = train_ds.map(
        lambda ex: prepare_train_features(ex, tok, max_length=max_length, doc_stride=doc_stride),
        batched=True,
        remove_columns=train_ds.column_names
    )

    val_feats = eval_ds.map(
        lambda ex: prepare_validation_features(ex, tok, max_length=max_length, doc_stride=doc_stride),
        batched=True,
        remove_columns=eval_ds.column_names
    )

    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name, cache_dir=cache_dir, local_files_only=offline
    )

    args = TrainingArguments(
        output_dir=os.path.join(out_dir, f"seed{seed}" + ("_D" if use_distill else "")),
        learning_rate=lr,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        num_train_epochs=epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="no",
        report_to=[],
        seed=seed,
    )

    metric = evaluate.load("squad")

    def postprocess_and_compute(p):
        raw = p.predictions
        preds = postprocess_qa_predictions(tok, eval_ds, val_feats, raw)
        formatted_preds = [{"id": k, "prediction_text": v} for k, v in preds.items()]
        refs = [{"id": ex["id"], "answers": ex["answers"]} for ex in eval_ds]
        out = metric.compute(predictions=formatted_preds, references=refs)
        return float(out["exact_match"]), float(out["f1"])

    if use_distill:
        if teacher_name is None:
            raise ValueError("teacher_name required when --distill is set")
        teacher = AutoModelForQuestionAnswering.from_pretrained(
            teacher_name, cache_dir=cache_dir, local_files_only=offline
        )
        trainer = DistillQATrainer(
            teacher=teacher,
            temperature=temperature,
            beta=beta,
            model=model,
            args=args,
            train_dataset=train_feats,
            eval_dataset=val_feats,
        )
    else:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_feats,
            eval_dataset=val_feats,
        )

    trainer.train()
    p = trainer.predict(val_feats)
    return postprocess_and_compute(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="eval_out/squad")
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46")

    ap.add_argument("--distill", action="store_true", help="enable DistilBERT(D) fine-tune distillation")
    ap.add_argument("--teacher", type=str, default=None, help="QA teacher model (for --distill)")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=1.0)

    # speed/memory knobs
    ap.add_argument("--max_train_samples", type=int, default=None, help="limit train examples (before tokenization)")
    ap.add_argument("--max_eval_samples", type=int, default=None, help="limit eval examples (before tokenization)")

    # tokenization knobs
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--doc_stride", type=int, default=128)

    # training knobs
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--train_bs", type=int, default=8)
    ap.add_argument("--eval_bs", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=2.0)

    ap.add_argument("--cache_dir", type=str, default="data/hf_cache")
    ap.add_argument("--offline", action="store_true", help="use only local HF cache (no downloads)")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    runs = []
    for s in seeds:
        em, f1 = run_one(
            model_name=args.model,
            seed=s,
            out_dir=args.out_dir,
            use_distill=args.distill,
            teacher_name=args.teacher,
            temperature=args.temperature,
            beta=args.beta,
            cache_dir=args.cache_dir,
            offline=args.offline,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            max_length=args.max_length,
            doc_stride=args.doc_stride,
            lr=args.lr,
            train_bs=args.train_bs,
            eval_bs=args.eval_bs,
            epochs=args.epochs,
        )
        runs.append((em, f1))
        tag = "D" if args.distill else "base"
        print(f"SQuAD {tag} seed={s}: EM={em:.2f} F1={f1:.2f}")

    ems = [x[0] for x in runs]
    f1s = [x[1] for x in runs]
    em_med = float(median(ems))
    f1_med = float(median(f1s))

    save_json(
        {
            "model": args.model,
            "teacher": args.teacher,
            "distill": args.distill,
            "seeds": seeds,
            "runs": [{"seed": s, "em": runs[i][0], "f1": runs[i][1]} for i, s in enumerate(seeds)],
            "median": {"em": em_med, "f1": f1_med},
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "max_length": args.max_length,
            "doc_stride": args.doc_stride,
            "lr": args.lr,
            "train_bs": args.train_bs,
            "eval_bs": args.eval_bs,
            "epochs": args.epochs,
            "cache_dir": args.cache_dir,
            "offline": args.offline
        },
        os.path.join(args.out_dir, "squad.json")
    )

    # naming for table
    label = "DistilBERT(D)" if args.distill else "DistilBERT"
    if args.model == "bert-base-uncased":
        label = "BERT-base"

    print_table(
        "TABLE (SQuAD v1.1 dev EM/F1)",
        ["model", "EM_median", "F1_median", "runs"],
        [[label, f"{em_med:.2f}", f"{f1_med:.2f}", str([(round(a, 2), round(b, 2)) for a, b in runs])]]
    )


if __name__ == "__main__":
    main()
