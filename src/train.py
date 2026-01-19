import math
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from dataset import DistillationDataset
from model import DistilBertStudent


def masked_kl_div(student_logits, teacher_logits, labels, temperature=2.0):
    """
    KL(student || teacher) on masked positions only.
    labels: [B, L] with -100 for non-MLM positions.
    """
    mask = labels != -100  # [B, L]
    if mask.sum().item() == 0:
        return torch.tensor(0.0, device=student_logits.device)

    # Select masked token positions: [N, V]
    s = student_logits[mask]
    t = teacher_logits[mask]

    T = temperature
    s_log_probs = F.log_softmax(s / T, dim=-1)
    t_probs = F.softmax(t / T, dim=-1)

    # Hinton distillation typically multiplies by T^2
    loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T * T)
    return loss


def cosine_hidden_loss(student_hidden_states, teacher_hidden_states, attention_mask):
    """
    Align hidden state directions using cosine loss.
    We map student layer j to teacher layer 2*j (including embeddings at index 0).
    student_hidden_states: tuple length 7 (emb + 6 layers)
    teacher_hidden_states: tuple length 13 (emb + 12 layers)
    attention_mask: [B, L]
    """
    # attention mask to float for weighting
    attn = attention_mask.float()  # [B, L]
    denom = attn.sum().clamp(min=1.0)

    total = 0.0
    n = 0

    # Map indices: 0..6 -> 0..12 step 2
    max_j = min(len(student_hidden_states), (len(teacher_hidden_states) + 1) // 2)
    for j in range(max_j):
        tj = 2 * j
        if tj >= len(teacher_hidden_states):
            break

        s_h = student_hidden_states[j]   # [B, L, H]
        t_h = teacher_hidden_states[tj]  # [B, L, H]

        # cosine similarity per token: [B, L]
        cos = F.cosine_similarity(s_h, t_h, dim=-1)

        # loss = 1 - cosine, masked by attention_mask
        layer_loss = ((1.0 - cos) * attn).sum() / denom
        total = total + layer_loss
        n += 1

    if n == 0:
        return torch.tensor(0.0, device=attention_mask.device)
    return total / n


def train(
    teacher_model_name="bert-base-uncased",
    batch_size=8,
    subset_size=10000,
    lr=5e-5,
    epochs=1,
    temperature=2.0,
    alpha_mlm=1.0,
    beta_distill=1.0,
    gamma_cos=1.0,
    max_steps=None,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    ds = DistillationDataset(model_name=teacher_model_name, subset_size=subset_size)
    loader = ds.get_data_loader(batch_size=batch_size)

    # Model
    model = DistilBertStudent(teacher_model_name=teacher_model_name)
    model.initialize_from_teacher()
    model.to(device)

    # Optimizer: student only (teacher frozen)
    optimizer = AdamW(model.student.parameters(), lr=lr)

    model.train()
    step = 0

    for epoch in range(epochs):
        for batch in loader:
            step += 1
            if max_steps is not None and step > max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Teacher forward (no grad)
            teacher_logits, teacher_hid = model.forward_teacher(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Student forward (with MLM loss)
            mlm_loss, student_logits, student_hid = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Distillation loss on masked tokens
            distill_loss = masked_kl_div(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                temperature=temperature
            )

            # Cosine loss on hidden states (masked by attention)
            cos_loss = cosine_hidden_loss(
                student_hidden_states=student_hid,
                teacher_hidden_states=teacher_hid,
                attention_mask=attention_mask
            )

            total_loss = alpha_mlm * mlm_loss + beta_distill * distill_loss + gamma_cos * cos_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.student.parameters(), 1.0)
            optimizer.step()

            if step % 20 == 0 or step == 1:
                print(
                    f"epoch={epoch+1} step={step} "
                    f"total={total_loss.item():.4f} "
                    f"mlm={mlm_loss.item():.4f} "
                    f"distill={distill_loss.item():.4f} "
                    f"cos={cos_loss.item():.4f}"
                )

        if max_steps is not None and step > max_steps:
            break

    print("Training done.")
    return model


if __name__ == "__main__":
    train(
        epochs=3,
        batch_size=8,
        subset_size=10000,
        lr=5e-5,
        temperature=2.0,
        alpha_mlm=1.0,
        beta_distill=1.0,
        gamma_cos=1.0,
        max_steps=2000
    )
