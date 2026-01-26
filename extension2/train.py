import torch
import torch.nn.functional as F
from torch.optim import AdamW
from dataset import DistillationDataset
from model import DistilBertStudent

def masked_kl_div(student_logits, teacher_logits, labels, temperature=2.0):
    mask = labels != -100
    if mask.sum().item() == 0: return torch.tensor(0.0, device=student_logits.device)
    s = student_logits[mask]
    t = teacher_logits[mask]
    T = temperature
    loss = F.kl_div(F.log_softmax(s/T, dim=-1), F.softmax(t/T, dim=-1), reduction="batchmean") * (T * T)
    return loss

def attention_distillation_loss(student_atts, teacher_atts):
    """ MSE between Student [6 layers] and Teacher [mapped 12 layers] attentions """
    mse_loss = torch.nn.MSELoss()
    total_attn_loss = 0.0
    for i in range(6):
        # We map student layer i to teacher layer (2i + 1)
        s_att = student_atts[i]
        t_att = teacher_atts[(i * 2) + 1]
        total_attn_loss += mse_loss(s_att, t_att)
    return total_attn_loss / 6

def train(epochs=3, batch_size=8, subset_size=10000, lr=5e-5, delta_attn=2.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = DistillationDataset(subset_size=subset_size)
    loader = ds.get_data_loader(batch_size=batch_size)
    
    model = DistilBertStudent()
    model.initialize_from_teacher()
    model.to(device)
    
    optimizer = AdamW(model.student.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward Passes
            t_logits, t_hid, t_atts = model.forward_teacher(input_ids, attention_mask)
            mlm_loss, s_logits, s_hid, s_atts = model(input_ids, attention_mask, labels)

            # Distillation Losses
            distill_loss = masked_kl_div(s_logits, t_logits, labels)
            attn_loss = attention_distillation_loss(s_atts, t_atts)

            # Final Objective
            total_loss = mlm_loss + distill_loss + (delta_attn * attn_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            #if step % 50 == 0:
            #    print(f"Epoch {epoch} | Step {step} | Total: {total_loss:.4f} | Attn: {attn_loss:.4f}")

            if step % 1 == 0:  # Print every single step instead of every 50
                print(f"Step {step} | Total: {total_loss:.4f} | Attn: {attn_loss:.4f} | MLM: {mlm_loss:.4f}")

if __name__ == "__main__":
    train(
        epochs=1,            # Just 1 pass
        batch_size=4,        # Smaller batches for quick updates
        subset_size=100,     # <--- Take only 100 sentences from the file
        lr=5e-5,
        delta_attn=2.0
    )

    # train ()