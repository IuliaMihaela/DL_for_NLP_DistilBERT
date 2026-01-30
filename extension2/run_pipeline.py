import os
import subprocess
import argparse
from datetime import datetime

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

def run(cmd, cwd=None):
    if cwd is None:
        cwd = SRC_DIR
    print(f"\n>> [CWD: {cwd}]", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)

def eval_all(model_path, out_root):
    os.makedirs(out_root, exist_ok=True)
    eval_scripts = [
        ("eval/eval_glue.py", "glue"),
        ("eval/eval_imdb.py", "imdb"),
        ("eval/eval_squad.py", "squad"),
        ("eval/benchmark_speed.py", None)
    ]
    for script_rel_path, sub_dir in eval_scripts:
        full_script_path = os.path.join(SRC_DIR, script_rel_path)
        if not os.path.exists(full_script_path):
            print(f"!! Warning: Skipping {script_rel_path}")
            continue
        cmd = ["python", script_rel_path, "--model", model_path]
        if sub_dir:
            cmd.extend(["--out_dir", os.path.join(out_root, sub_dir)])
        try:
            run(cmd)
        except subprocess.CalledProcessError:
            print(f"!! Error running {script_rel_path}")

def train_and_eval(tag, dataset_mode, subset_size, out_model_dir, eval_dir,
                   data_dir, epochs, batch_size, grad_accum, delta_attn, no_fp16=False):
    
    # 1. TRAIN
    cmd = ["python", "train.py",
           "--epochs", str(epochs),
           "--batch_size", str(batch_size),
           "--subset_size", str(subset_size),
           "--grad_accum_steps", str(grad_accum),
           "--delta_attn", str(delta_attn),  # This passes the weight to train.py
           "--dataset_mode", dataset_mode,
           "--output_dir", out_model_dir,
           "--save_dir", os.path.join(out_model_dir, "_steps")]
    
    if no_fp16:
        cmd.append("--no_fp16")
    if data_dir:
        cmd += ["--data_dir", data_dir]
    
    run(cmd)

    # 2. EVAL
    eval_all(out_model_dir, eval_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper_data_dir", type=str, default=None)
    ap.add_argument("--small_subset", type=int, default=2000)
    ap.add_argument("--paper_subset", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--delta_attn", type=float, default=1.0) # Added argument
    ap.add_argument("--separate_only", action="store_true")
    ap.add_argument("--small_ckpt", type=str, default="checkpoints_small/final")
    ap.add_argument("--paper_ckpt", type=str, default="checkpoints_paper/final")
    ap.add_argument("--out_root", type=str, default="runs")
    ap.add_argument("--no_fp16", action="store_true")

    args = ap.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(args.out_root, stamp)
    os.makedirs(root, exist_ok=True)

    if args.separate_only:
        if os.path.isdir(args.small_ckpt):
            eval_all(args.small_ckpt, os.path.join(root, "EVAL_existing_small"))
        return

    # --- Run 1: TRAIN SMALL ---
    train_and_eval(
        tag="TRAIN_small",
        dataset_mode="small",
        subset_size=args.small_subset,
        out_model_dir=os.path.join(root, "CKPT_small"),
        eval_dir=os.path.join(root, "TABLES_small"),
        data_dir=None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        delta_attn=args.delta_attn, # Included here
        no_fp16=args.no_fp16
    )

    # --- Run 2: TRAIN PAPER ---
    train_and_eval(
        tag="TRAIN_paper",
        dataset_mode="paper",
        subset_size=args.paper_subset,
        out_model_dir=os.path.join(root, "CKPT_paper"),
        eval_dir=os.path.join(root, "TABLES_paper"),
        data_dir=args.paper_data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        delta_attn=args.delta_attn, # Included here
        no_fp16=args.no_fp16
    )

if __name__ == "__main__":
    main()