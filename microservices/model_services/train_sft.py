# microservices/model_service/train_sft.py

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os

def save_checkpoint(model, optimizer, epoch, avg_train_loss, checkpoint_dir="./checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    file_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'avg_train_loss': avg_train_loss,
    }, file_path)
    print(f"[Checkpoint] Saved: {file_path}")



def load_checkpoint(checkpoint_file, model, optimizer):
    """
    Loads model/optimizer states and the epoch from a checkpoint file.
    """
    print(f"[Checkpoint] Loading from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    print(f"[Checkpoint] Resumed from epoch {start_epoch}")
    return start_epoch



def evaluate_sft(model, dataloader, device="cuda"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def train_sft(model, 
              train_ds, 
              test_ds,
              device="cuda", 
              epochs=3, 
              batch_size=16, 
              lr=2e-5,
              checkpoint_dir="./checkpoints",
              resume=True,
              resume_checkpoint="./checkpoints/model_epoch_1.pt"):
    """
    Fine-tunes the classification model (SFT) with optional checkpointing.
    
    Args:
      checkpoint_dir (str): Directory to save checkpoints. If None, no checkpoints are saved.
      resume (bool): Whether to resume from a given checkpoint file.
      resume_checkpoint (str): Path to a checkpoint file to resume from.
    """
    # 1. Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 2. Prepare model & optimizer
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # 3. If resuming, load checkpoint
    start_epoch = 0
    if resume and resume_checkpoint is not None and os.path.isfile(resume_checkpoint):
        start_epoch = load_checkpoint(resume_checkpoint, model, optimizer)

    # 4. Train loop
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # IMPORTANT: use loss.item() not loss.items()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_acc = evaluate_sft(model, test_loader, device)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 5. Save checkpoint after each epoch (if checkpoint_dir is provided)
        if checkpoint_dir is not None:
            save_checkpoint(model, optimizer, epoch+1, avg_train_loss, checkpoint_dir)

    return model
