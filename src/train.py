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



def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_kl_div(student_logits, teacher_logits, labels, temperature=2.0):
    """
    Distillation loss on MLM masked positions only.
    Uses full teacher distribution over vocab at those positions.
    """
    mask = labels != -100  # [B, L]
    if mask.sum().item() == 0:
        return torch.tensor(0.0, device=student_logits.device)

    s = student_logits[mask]  # [N_masked, V]
    t = teacher_logits[mask]  # [N_masked, V]

    T = float(temperature)
    s_log_probs = F.log_softmax(s / T, dim=-1)
    t_probs = F.softmax(t / T, dim=-1)

    # Hinton: multiply by T^2
    return F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T * T)

def full_distillation_loss(student_logits, teacher_logits, attention_mask, temperature=2.0):
    """
    KL(student || teacher) on ALL tokens (masked + unmasked), ignoring padding.
    """
    # Flatten to [Batch * SeqLen, Vocab]
    vocab_size = student_logits.size(-1)
    s_logits_flat = student_logits.view(-1, vocab_size)
    t_logits_flat = teacher_logits.view(-1, vocab_size)
    
    # Create mask for valid tokens (1 = token, 0 = padding)
    # Flatten attention_mask to [Batch * SeqLen]
    mask_flat = attention_mask.view(-1) == 1
    
    # Select only valid tokens (remove padding)
    s_logits_active = s_logits_flat[mask_flat]
    t_logits_active = t_logits_flat[mask_flat]
    
    T = float(temperature)
    
    # Log Softmax for Student (KL input)
    s_log_probs = F.log_softmax(s_logits_active / T, dim=-1)
    
    # Softmax for Teacher (KL target)
    t_probs = F.softmax(t_logits_active / T, dim=-1)
    
    # KL * T^2
    return F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T * T)


def cosine_hidden_loss_last(student_hidden_states, teacher_hidden_states, attention_mask):
    """
    Cosine alignment on last hidden states only.
    """
    attn = attention_mask.float()  # [B, L]
    denom = attn.sum().clamp(min=1.0)

    s_h = student_hidden_states[-1]  # [B, L, H]
    t_h = teacher_hidden_states[-1]  # [B, L, H]

    cos = F.cosine_similarity(s_h, t_h, dim=-1)  # [B, L]
    return ((1.0 - cos) * attn).sum() / denom

def cosine_loss_all_layers(student_hidden_states, teacher_hidden_states, attention_mask):
    """
    Aligns all corresponding hidden states (Student i -> Teacher 2*i).
    """
    attn = attention_mask.float()  # [B, L]
    denom = attn.sum().clamp(min=1.0)
    
    total_loss = 0.0
    n_layers_aligned = 0
    
    # Iterate over student layers
    # student_hidden_states is a tuple of length (n_layers + 1) for embeddings
    for i, s_h in enumerate(student_hidden_states):
        teacher_idx = i * 2
        
        # Ensure we don't go out of bounds of the teacher
        if teacher_idx >= len(teacher_hidden_states):
            break
            
        t_h = teacher_hidden_states[teacher_idx]
        
        # Calculate cosine similarity for this layer pair
        cos = F.cosine_similarity(s_h, t_h, dim=-1)  # [B, L]
        
        # Loss = 1 - cosine (masked by attention)
        layer_loss = ((1.0 - cos) * attn).sum() / denom
        
        total_loss += layer_loss
        n_layers_aligned += 1
        
    if n_layers_aligned == 0:
        return torch.tensor(0.0, device=student_hidden_states[0].device)
        
    return total_loss / n_layers_aligned

@dataclass
class TrainCfg:
    teacher_model_name: str = "bert-base-uncased"
    batch_size: int = 8
    subset_size: int = 10000
    epochs: int = 3

    lr: float = 5e-5
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-6

    warmup_ratio: float = 0.06
    max_steps: int | None = None  # optimizer steps
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0

    temperature: float = 2.0
    alpha_mlm: float = 1.0
    beta_distill: float = 1.0
    gamma_cos: float = 1.0

    fp16: bool = True
    seed: int = 42

    save_dir: str = "checkpoints"       # for intermediate step_* checkpoints
    save_every_steps: int = 500         # optimizer steps
    log_every_steps: int = 50           # optimizer steps

    dataset_mode: str = "small"         # "small" or "paper"
    data_dir: str | None = None
    output_dir: str = "checkpoints/final"  # final model dir


def train(cfg: TrainCfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = bool(cfg.fp16 and device == "cuda")

    set_all_seeds(cfg.seed)

   
    ds = DistillationDataset(
        model_name=cfg.teacher_model_name,
        subset_size=cfg.subset_size,
        mode=cfg.dataset_mode,
        data_dir=cfg.data_dir
    )
    loader = ds.get_data_loader(batch_size=cfg.batch_size)


    model = DistilBertStudent(teacher_model_name=cfg.teacher_model_name)
    model.initialize_from_teacher()
    model.to(device)

    model.teacher.eval()
    model.student.train()


    optimizer = AdamW(
        model.student.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

        # Calculate total training steps
    loader = ds.get_data_loader(batch_size=cfg.batch_size)
    try:
        # Try getting the exact length 
        total_batches = len(loader)
    except TypeError:
        # If streaming (WikiText-103), we use an estimate.
        print("!! Streaming dataset detected (no len()). Using estimated size for progress bar.")
        estimated_samples = 781000 
        total_batches = estimated_samples // cfg.batch_size
    
    steps_per_epoch = math.ceil(total_batches / cfg.grad_accum_steps)


    total_optim_steps = steps_per_epoch * cfg.epochs

    # If user sets max_steps, cap training
    if cfg.max_steps is not None:
        total_optim_steps = min(total_optim_steps, cfg.max_steps)

    warmup_steps = int(cfg.warmup_ratio * total_optim_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optim_steps
    )

    # AMP scaler
    scaler = torch.amp.GradScaler(enabled=use_fp16)

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Train loop
    raw_step = 0     # batch counter
    optim_step = 0   # optimizer update counter

    running = {"mlm": 0.0, "distill": 0.0, "cos": 0.0, "total": 0.0}

    optimizer.zero_grad(set_to_none=True)

    stop = False
    for epoch in range(cfg.epochs):
        for batch in loader:
            raw_step += 1

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Teacher forward 
            teacher_logits, teacher_hid = model.forward_teacher(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Student forward + losses
            with torch.amp.autocast(enabled=use_fp16):
                mlm_loss, student_logits, student_hid = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                # distill_loss = masked_kl_div(
                #     student_logits=student_logits,
                #     teacher_logits=teacher_logits,
                #     labels=labels,
                #     temperature=cfg.temperature
                # )

                # cos_loss = cosine_hidden_loss_last(
                #     student_hidden_states=student_hid,
                #     teacher_hidden_states=teacher_hid,
                #     attention_mask=attention_mask
                # )
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

                total_loss = (
                    cfg.alpha_mlm * mlm_loss +
                    cfg.beta_distill * distill_loss +
                    cfg.gamma_cos * cos_loss
                )

                # scale for grad accumulation
                loss_scaled = total_loss / cfg.grad_accum_steps

            # backward
            scaler.scale(loss_scaled).backward()

            # if accumulation boundary reached, do optimizer step
            if (raw_step % cfg.grad_accum_steps) == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.student.parameters(), cfg.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                optim_step += 1

                # track logs on optimizer step
                running["mlm"] += float(mlm_loss.detach().cpu())
                running["distill"] += float(distill_loss.detach().cpu())
                running["cos"] += float(cos_loss.detach().cpu())
                running["total"] += float(total_loss.detach().cpu())

                if optim_step % cfg.log_every_steps == 0:
                    denom = cfg.log_every_steps
                    print(
                        f"epoch={epoch+1} optim_step={optim_step}/{total_optim_steps} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} "
                        f"total={running['total']/denom:.4f} "
                        f"mlm={running['mlm']/denom:.4f} "
                        f"distill={running['distill']/denom:.4f} "
                        f"cos={running['cos']/denom:.4f}"
                    )
                    running = {"mlm": 0.0, "distill": 0.0, "cos": 0.0, "total": 0.0}

                if cfg.save_every_steps and (optim_step % cfg.save_every_steps == 0):
                    ckpt_dir = os.path.join(cfg.save_dir, f"step_{optim_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    model.student.save_pretrained(ckpt_dir)
                    ds.tokenizer.save_pretrained(ckpt_dir)
                    print(f"Saved checkpoint: {ckpt_dir}")

                if cfg.max_steps is not None and optim_step >= cfg.max_steps:
                    stop = True
                    break

        if stop:
            break

    # flush leftover grads if epoch ended mid-accumulation
    # and if still steps left 
    if (raw_step % cfg.grad_accum_steps) != 0 and (cfg.max_steps is None or optim_step < cfg.max_steps):
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.student.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        optim_step += 1

    # final save
    model.student.save_pretrained(cfg.output_dir)
    ds.tokenizer.save_pretrained(cfg.output_dir)
    print(f"Training done. Final model saved to: {cfg.output_dir}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--subset_size", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=0, help="optimizer steps; 0 disables")
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha_mlm", type=float, default=1.0)
    ap.add_argument("--beta_distill", type=float, default=1.0)
    ap.add_argument("--gamma_cos", type=float, default=1.0)
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument("--output_dir", type=str, default="checkpoints/final")

    ap.add_argument("--dataset_mode", type=str, default="small", choices=["small", "paper"])
    ap.add_argument("--data_dir", type=str, default="", help="required for dataset_mode=paper")

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
        fp16=(not args.no_fp16),
        seed=args.seed,
        save_dir=args.save_dir,
        output_dir=args.output_dir,
        dataset_mode=args.dataset_mode,
        data_dir=(args.data_dir if args.data_dir else None),
    )

    # paper mode needs data_dir (we can as well not use it)
    if cfg.dataset_mode == "paper" and cfg.data_dir is None:
        raise ValueError("dataset_mode='paper' requires --data_dir")

    train(cfg)


if __name__ == "__main__":
    main()
