# src/training.py

import torch
import math
from torch.utils.data import DataLoader
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

def distillation_loss_fn(student_logits, teacher_logits, labels, alpha_ce=0.5, alpha_mle=0.5, temperature=2.0):
    """
    Combines:
    - Soft loss: KL divergence between teacher's distribution and student's
    - Hard loss: standard cross-entropy vs. the ground-truth labels

    alpha_ce + alpha_mle = 1.0 is typical
    """
    # Soft loss: KL Divergence
    # teacher_probs = softmax(teacher_logits / T), student_probs = softmax(student_logits / T)
    # For efficiency, we use the built-in PyTorch method with log_softmax
    log_probs_student = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
    probs_teacher = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)

    kl_div = torch.nn.functional.kl_div(
        log_probs_student, 
        probs_teacher, 
        reduction="batchmean"
    ) * (temperature**2)

    # Hard loss: cross-entropy with labels
    # Note: ensure ignore_index=-100 for padded labels
    ce_loss = torch.nn.functional.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )

    loss = alpha_ce * kl_div + alpha_mle * ce_loss
    return loss, kl_div, ce_loss

def train_distillation(
    teacher_model: LlamaForCausalLM,
    student_model: LlamaForCausalLM,
    train_dataset,
    tokenizer,
    output_dir="output-distilled",
    epochs=1,
    batch_size=2,
    lr=1e-5,
    warmup_ratio=0.1,
    alpha_ce=0.5,
    alpha_mle=0.5,
    temperature=2.0
):
    """
    Distills the teacher model knowledge into the student model using
    a combined cross-entropy + KL-distillation loss.
    """

    # Freeze teacher model
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Prepare data loader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Set up optimizer, scheduler
    optimizer = AdamW(student_model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * total_steps),
        num_training_steps=total_steps
    )

    student_model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model.to(device)
    teacher_model.to(device)

    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 1) Teacher forward (no gradient)
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            teacher_logits = teacher_outputs.logits

            # 2) Student forward
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            student_logits = student_outputs.logits

            # 3) Distillation loss
            loss, kd_loss, mle_loss = distillation_loss_fn(
                student_logits, teacher_logits, labels,
                alpha_ce=alpha_ce,
                alpha_mle=alpha_mle,
                temperature=temperature
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")

    # Save the student model
    student_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Distilled model saved to {output_dir}")
