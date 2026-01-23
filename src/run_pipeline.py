import os
import subprocess
import argparse
from datetime import datetime

# Get the absolute path of the 'src' folder
# This ensures train.py is found even if you run the script from the project root
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

def run(cmd, cwd=None):
    # If no cwd is specified, run inside the src folder
    if cwd is None:
        cwd = SRC_DIR
    print(f"\n>> [CWD: {cwd}]", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)

def eval_all(model_path: str, out_root: str):
    """
    Runs all evaluation scripts found in the eval/ subdirectory.
    """
    os.makedirs(out_root, exist_ok=True)

    # Define the mapping of script names to output folders
    # These paths are relative to the src/ folder
    eval_scripts = [
        ("eval/eval_glue.py", "glue"),
        ("eval/eval_imdb.py", "imdb"),
        ("eval/eval_squad.py", "squad"),
        ("eval/benchmark_speed.py", None) # No output dir needed for speed benchmark
    ]

    for script_rel_path, sub_dir in eval_scripts:
        # Check if the script exists inside src/ before trying to run it
        full_script_path = os.path.join(SRC_DIR, script_rel_path)
        if not os.path.exists(full_script_path):
            print(f"!! Warning: Skipping {script_rel_path} (File not found at {full_script_path})")
            continue
        
        cmd = ["python", script_rel_path, "--model", model_path]
        if sub_dir:
            cmd.extend(["--out_dir", os.path.join(out_root, sub_dir)])
        
        try:
            run(cmd)
        except subprocess.CalledProcessError:
            print(f"!! Error running {script_rel_path}, continuing to next eval...")

def train_and_eval(tag: str, dataset_mode: str, subset_size: int, out_model_dir: str, eval_dir: str,
                   data_dir: str | None, epochs: int, batch_size: int, grad_accum: int, no_fp16: bool = False):
    
    # 1. TRAIN
    cmd = ["python", "train.py",
           "--epochs", str(epochs),
           "--batch_size", str(batch_size),
           "--subset_size", str(subset_size),
           "--grad_accum_steps", str(grad_accum),
           "--dataset_mode", dataset_mode,
           "--output_dir", out_model_dir,
           "--save_dir", os.path.join(out_model_dir, "_steps")]
    
    if no_fp16:
        cmd.append("--no_fp16")

    # Only add data_dir if it's actually provided
    if data_dir:
        cmd += ["--data_dir", data_dir]
    
    run(cmd)

    # 2. EVAL
    eval_all(out_model_dir, eval_dir)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper_data_dir", type=str, default=None, help="path to prepared paper corpus")
    
    ap.add_argument("--small_subset", type=int, default=2000)
    ap.add_argument("--paper_subset", type=int, default=0, help="0 = use full paper ds")
    
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    
    ap.add_argument("--separate_only", action="store_true", help="only run eval on existing checkpoints")
    
    # Checkpoints paths
    ap.add_argument("--small_ckpt", type=str, default="checkpoints_small/final")
    ap.add_argument("--paper_ckpt", type=str, default="checkpoints_paper/final")
    ap.add_argument("--out_root", type=str, default="runs")
    
    ap.add_argument("--no_fp16", action="store_true", help="Disable mixed precision (fixes NaN for small batches)")

    args = ap.parse_args()

    # Create run directory
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(args.out_root, stamp)
    os.makedirs(root, exist_ok=True)

    # --- Run 3 & 4: Eval Existing Checkpoints (if found) ---
    if os.path.isdir(args.small_ckpt):
        print(f"Found existing small ckpt at {args.small_ckpt}, evaluating...")
        eval_all(args.small_ckpt, os.path.join(root, "EVAL_existing_small"))
    
    if os.path.isdir(args.paper_ckpt):
        print(f"Found existing paper ckpt at {args.paper_ckpt}, evaluating...")
        eval_all(args.paper_ckpt, os.path.join(root, "EVAL_existing_paper"))

    if args.separate_only:
        return

    # --- Run 1: Train Small -> Eval ---
    print("\n=== Starting Run 1: TRAIN SMALL ===")
    train_and_eval(
        tag="TRAIN_small",
        dataset_mode="small",
        subset_size=args.small_subset,
        out_model_dir=os.path.join(root, "CKPT_small"),
        eval_dir=os.path.join(root, "TABLES_small"),
        data_dir=None, # Small mode generates its own data
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        no_fp16=args.no_fp16
    )

    # --- Run 2: Train Paper -> Eval ---
    print("\n=== Starting Run 2: TRAIN PAPER ===")
    
    if args.paper_data_dir is None:
         print("!! Warning: No --paper_data_dir provided. If train.py requires it, this will crash.")
    
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
        no_fp16=args.no_fp16
    )

if __name__ == "__main__":
    main()