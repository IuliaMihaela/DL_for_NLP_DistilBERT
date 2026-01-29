import os
import json
import random
import numpy as np
import torch

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    if n % 2 == 1:
        return float(xs[n//2])
    return float((xs[n//2 - 1] + xs[n//2]) / 2.0)

def print_table(title: str, headers, rows):
    print("\n" + title)
    colw = [len(h) for h in headers]
    for r in rows:
        for i, v in enumerate(r):
            colw[i] = max(colw[i], len(str(v)))
    fmt = " | ".join("{:<" + str(w) + "}" for w in colw)
    print(fmt.format(*headers))
    print("-+-".join("-"*w for w in colw))
    for r in rows:
        print(fmt.format(*r))