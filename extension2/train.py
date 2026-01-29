import os
import math
import random
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from dataset import DistillationDataset
from model import DistilBertStudent


# ---------------------------
# utils
# ---------------------------

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------
# goal: minimize the difference between the Teacher’s attention distribution and the Student’s
#       using Kullback-Leibler(KL) Divergence (std. way to measure the difference b/W probability distribution & 
#       reference distribution)
# Teaches the process (how to look at the sentence)
# --------------------------- 
# ATTENTION MAP DISTILLATION
def attention_map_distillation_loss(student_atts, teacher_atts, attention_mask):
    
    # reshaping the mask so it can be multiplied by the 4D attention maps 
    # --> only ant to calculate loss on "real" tokens, not padding tokens
    mask = attention_mask.unsqueeze(1).unsqueeze(2).float()
    
    total_loss = 0.0
    n_layers = 0
    
    # Layer Mapping
    for i, s_att in enumerate(student_atts):
        # Student layer i matches Teacher layer 2i + 1 
        teacher_idx = (i * 2) + 1 
        if teacher_idx >= len(teacher_atts):
            break
        
        t_att = teacher_atts[teacher_idx]

        # Log-Probability Conversion (Converts the Student's attention scores into log-space)
        # add a tiny epsilon to avoid log(0)
        s_log_probs = torch.log(s_att + 1e-12)
        
        # KL Divergence: Teacher is the target (p), Student is the input (log q)
        # reduction="none" --> get a loss value for every single token relation before we apply the mask.
        kl_loss = F.kl_div(s_log_probs, t_att, reduction="none")
        
        # Masking(to ignore the paddings) & Normalizing(to ensure the loss remains stable)
        masked_kl = (kl_loss * mask).sum() / (mask.sum() * s_att.size(1)) # avg by heads
        total_loss += masked_kl
        n_layers += 1

    # returns a single scalar tensor representing the mean KL Divergence across all paired layers and tokens.
    return total_loss / n_layers if n_layers > 0 else torch.tensor(0.0, device=attention_mask.device)

# ---------------------------
# goal: soft target 
# Teaches the result (what the final answer should look like)
# ---------------------------
# LOGIT DISTILLATION 
def full_distillation_loss(student_logits, teacher_logits, attention_mask, temperature=2.0):
    
    # Reshaping for Calculation; Flatten [Batch, Seq\_Len, Vocab] --> [Batch * SeqLen, Vocab]
    vocab_size = student_logits.size(-1)
    s_logits_flat = student_logits.view(-1, vocab_size)
    t_logits_flat = teacher_logits.view(-1, vocab_size)
    mask_flat = attention_mask.view(-1) == 1
    
    # Boolean Masking --> filters out the padding tokens
    # s_logits_active only contains the predictions for actual words
    s_logits_active = s_logits_flat[mask_flat]
    t_logits_active = t_logits_flat[mask_flat]
    
    # Temperature Scaling (T); to reveal the "hidden knowledge"
    # Capturing "Dark Knowledge" ; soften the curve
    T = float(temperature)
    s_log_probs = F.log_softmax(s_logits_active / T, dim=-1)
    t_probs = F.softmax(t_logits_active / T, dim=-1)
    
    # KL Divergence and Temp. Scaling 
    return F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T * T)

# ---------------------------
# intermediate representations: ensures that the vectors the Student model creates 
# at each layer point is in the same "semantic direction" as the vectors the Teacher creates.
# ---------------------------
# Hidden State Alignment using Cosine Distance
def cosine_loss_all_layers(student_hidden_states, teacher_hidden_states, attention_mask):
    """
    Aligns all corresponding hidden states (Student i -> Teacher 2*i).
    """
    # Normalization Setup
    attn = attention_mask.float()
    denom = attn.sum().clamp(min=1.0)
    total_loss = 0.0
    n_layers_aligned = 0

    # Iterate over student layers
    # student_hidden_states is a tuple of length (n_layers + 1) for embeddings
    for i, s_h in enumerate(student_hidden_states):
        teacher_idx = i * 2

        # Safety check to ensure we don't go out of bounds of the teacher
        if teacher_idx >= len(teacher_hidden_states):
            break
        t_h = teacher_hidden_states[teacher_idx]

         # Calculate cosine similarity for this layer pair
        cos = F.cosine_similarity(s_h, t_h, dim=-1) # [B, L]

        # Converting Similarity to Loss
        # Loss = 1 - cosine (masked by attention)
        layer_loss = ((1.0 - cos) * attn).sum() / denom

        total_loss += layer_loss
        n_layers_aligned += 1

    # returns a single scalar tensor representing the avg. cosine distance b/w student & teacher's hidden states across all aligned layers  
    return total_loss / n_layers_aligned if n_layers_aligned > 0 else torch.tensor(0.0, device=student_hidden_states[0].device)

# ---------------------------
# 
# ---------------------------
@dataclass
class TrainCfg:
    # Model & Data Basics
    teacher_model_name: str = "bert-base-uncased"
    batch_size: int = 8
    subset_size: int = 10000
    epochs: int = 3

    # Optimizer Settings (AdamW)
    lr: float = 5e-5
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-6

    # Training Dynamics
    warmup_ratio: float = 0.06
    max_steps: int = None  # Removed the | None
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0

    # Loss Weighting
    temperature: float = 2.0
    alpha_mlm: float = 1.0
    beta_distill: float = 1.0

    gamma_cos: float = 1.0
    delta_attn: float = 1.0

    # System & Logging
    fp16: bool = True
    seed: int = 42
    save_dir: str = "checkpoints"
    save_every_steps: int = 500
    log_every_steps: int = 50
    dataset_mode: str = "small"
    data_dir: str = None    # Removed the | None
    output_dir: str = "checkpoints/final"


# ---------------------------
# 
# ---------------------------
def train(cfg: TrainCfg):
    # Setup and Initialization
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = bool(cfg.fp16 and device == "cuda")
    set_all_seeds(cfg.seed)

    ds = DistillationDataset(
        model_name=cfg.teacher_model_name,
        subset_size=cfg.subset_size,
        mode=cfg.dataset_mode,
        data_dir=cfg.data_dir
    )

    # Data Preparation
    loader = ds.get_data_loader(batch_size=cfg.batch_size)

    model = DistilBertStudent(teacher_model_name=cfg.teacher_model_name)
    model.initialize_from_teacher()
    model.to(device)

    model.teacher.eval()
    model.student.train()

    # Optimization Logic
    optimizer = AdamW(
        model.student.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    steps_per_epoch = math.ceil(len(loader) / cfg.grad_accum_steps)
    total_optim_steps = steps_per_epoch * cfg.epochs
    if cfg.max_steps is not None:
        total_optim_steps = min(total_optim_steps, cfg.max_steps)

    warmup_steps = int(cfg.warmup_ratio * total_optim_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_optim_steps)
    # 
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)

    raw_step = 0
    optim_step = 0
    # Added "attn" to tracking
    running = {"mlm": 0.0, "distill": 0.0, "cos": 0.0, "attn": 0.0, "total": 0.0}

    optimizer.zero_grad(set_to_none=True)
    stop = False

    for epoch in range(cfg.epochs):
        for batch in loader:
            raw_step += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 1. Teacher forward (Now returns teacher_atts)
            teacher_logits, teacher_hid, teacher_atts = model.forward_teacher(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # 2. Student forward (Now returns student_atts)
            with torch.cuda.amp.autocast(enabled=use_fp16):
                mlm_loss, student_logits, student_hid, student_atts = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                distill_loss = full_distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    attention_mask=attention_mask,  
                    temperature=cfg.temperature
                )
                cos_loss = cosine_loss_all_layers(
                    student_hidden_states=student_hid,
                    teacher_hidden_states=teacher_hid,
                    attention_mask=attention_mask
                )
                
                # 3. Calculate Attention Loss
                attn_loss = attention_map_distillation_loss(
                    student_atts=student_atts,
                    teacher_atts=teacher_atts,
                    attention_mask=attention_mask
                )

                # 4. Total Weighted Loss
                total_loss = (
                    cfg.alpha_mlm * mlm_loss +
                    cfg.beta_distill * distill_loss +
                    cfg.gamma_cos * cos_loss +
                    cfg.delta_attn * attn_loss  
                )

                # scale for grad accumulation
                loss_scaled = total_loss / cfg.grad_accum_steps

            # backward
            scaler.scale(loss_scaled).backward()

            # if we reached accumulation boundary, do optimizer step
            if (raw_step % cfg.grad_accum_steps) == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.student.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optim_step += 1

                running["mlm"] += float(mlm_loss.detach().cpu())
                running["distill"] += float(distill_loss.detach().cpu())
                running["cos"] += float(cos_loss.detach().cpu())
                running["attn"] += float(attn_loss.detach().cpu()) # <--- Tracking
                running["total"] += float(total_loss.detach().cpu())

                if optim_step % cfg.log_every_steps == 0:
                    denom = cfg.log_every_steps
                    print(
                        f"epoch={epoch+1} optim_step={optim_step}/{total_optim_steps} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} "
                        f"total={running['total']/denom:.4f} "
                        f"mlm={running['mlm']/denom:.4f} "
                        f"distill={running['distill']/denom:.4f} "
                        f"cos={running['cos']/denom:.4f} "
                        f"attn={running['attn']/denom:.4f}" # <--- Logged
                    )
                    running = {"mlm": 0.0, "distill": 0.0, "cos": 0.0, "attn": 0.0, "total": 0.0}

                if cfg.save_every_steps and (optim_step % cfg.save_every_steps == 0):
                    ckpt_dir = os.path.join(cfg.save_dir, f"step_{optim_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    model.student.save_pretrained(ckpt_dir)
                    ds.tokenizer.save_pretrained(ckpt_dir)
                    print(f"Saved checkpoint: {ckpt_dir}")

                if cfg.max_steps is not None and optim_step >= cfg.max_steps:
                    stop = True
                    break
        if stop: break

    # Final cleanup logic remains same...
    model.student.save_pretrained(cfg.output_dir)
    ds.tokenizer.save_pretrained(cfg.output_dir)
    print(f"Training done. Final model saved to: {cfg.output_dir}")
    return model

# ---------------------------
# 
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--subset_size", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha_mlm", type=float, default=1.0)
    ap.add_argument("--beta_distill", type=float, default=1.0)
    ap.add_argument("--gamma_cos", type=float, default=1.0)
    ap.add_argument("--delta_attn", type=float, default=1.0) # <--- Added to CLI
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument("--output_dir", type=str, default="checkpoints/final")
    ap.add_argument("--dataset_mode", type=str, default="small", choices=["small", "paper"])
    ap.add_argument("--data_dir", type=str, default="")

    args = ap.parse_args()

    cfg = TrainCfg(
        epochs=args.epochs,
        batch_size=args.batch_size,
        subset_size=args.subset_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_accum_steps=args.grad_accum_steps,
        max_steps=(args.max_steps if args.max_steps > 0 else None),
        temperature=args.temperature,
        alpha_mlm=args.alpha_mlm,
        beta_distill=args.beta_distill,
        gamma_cos=args.gamma_cos,
        delta_attn=args.delta_attn, # <--- Passed to Cfg
        fp16=(not args.no_fp16),
        seed=args.seed,
        save_dir=args.save_dir,
        output_dir=args.output_dir,
        dataset_mode=args.dataset_mode,
        data_dir=(args.data_dir if args.data_dir else None),
    )

    if cfg.dataset_mode == "paper" and cfg.data_dir is None:
        raise ValueError("dataset_mode='paper' requires --data_dir")

    train(cfg)

if __name__ == "__main__":
    main()