import os
import subprocess
import argparse
from datetime import datetime

def run(cmd, cwd=None):
    print("\n>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)

def eval_all(model_path: str, out_root: str):
    os.makedirs(out_root, exist_ok=True)

    # Table 1 (GLUE)
    run(["python", "src/eval/eval_glue.py",
         "--model", model_path,
         "--out_dir", os.path.join(out_root, "glue")])

    # Table 2 (IMDb)
    run(["python", "src/eval/eval_imdb.py",
         "--model", model_path,
         "--out_dir", os.path.join(out_root, "imdb")])

    # Table 2 (SQuAD) base
    run(["python", "src/eval/eval_squad.py",
         "--model", model_path,
         "--out_dir", os.path.join(out_root, "squad")])

    # Table 3 (speed/params)
    run(["python", "src/eval/benchmark_speed.py",
         "--model", model_path])

def train_and_eval(tag: str, dataset_mode: str, subset_size: int, out_model_dir: str, eval_dir: str,
                   data_dir: str | None, epochs: int, batch_size: int, grad_accum: int):
    # train
    cmd = ["python", "src/train.py",
           "--epochs", str(epochs),
           "--batch_size", str(batch_size),
           "--subset_size", str(subset_size),
           "--grad_accum_steps", str(grad_accum),
           "--dataset_mode", dataset_mode,
           "--output_dir", out_model_dir,
           "--save_dir", os.path.join(out_model_dir, "_steps")]
    if data_dir:
        cmd += ["--data_dir", data_dir]
    run(cmd)

    # eval
    eval_all(out_model_dir, eval_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper_data_dir", type=str, default="", help="path to prepared paper corpus (optional)")
    ap.add_argument("--small_subset", type=int, default=2000)
    ap.add_argument("--paper_subset", type=int, default=0, help="0 = use full paper ds, otherwise subset")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--separate_only", action="store_true",
                    help="only run eval on existing checkpoints (no training)")
    ap.add_argument("--small_ckpt", type=str, default="checkpoints_small/final")
    ap.add_argument("--paper_ckpt", type=str, default="checkpoints_paper/final")
    ap.add_argument("--out_root", type=str, default="runs")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(args.out_root, stamp)
    os.makedirs(root, exist_ok=True)

    # Run 3: eval existing small checkpoint
    if os.path.isdir(args.small_ckpt):
        eval_all(args.small_ckpt, os.path.join(root, "EVAL_existing_small"))

    # Run 4: eval existing paper checkpoint
    if os.path.isdir(args.paper_ckpt):
        eval_all(args.paper_ckpt, os.path.join(root, "EVAL_existing_paper"))

    if args.separate_only:
        return

    # Run 1: train small -> eval
    train_and_eval(
        tag="TRAIN_small",
        dataset_mode="small",
        subset_size=args.small_subset,
        out_model_dir=os.path.join(root, "CKPT_small"),
        eval_dir=os.path.join(root, "TABLES_small"),
        data_dir=None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum
    )

    # Run 2: train paper -> eval
    paper_dir = args.paper_data_dir if args.paper_data_dir else None
    train_and_eval(
        tag="TRAIN_paper",
        dataset_mode="paper",
        subset_size=args.paper_subset,
        out_model_dir=os.path.join(root, "CKPT_paper"),
        eval_dir=os.path.join(root, "TABLES_paper"),
        data_dir=paper_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum
    )

if __name__ == "__main__":
    main()
