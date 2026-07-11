import os
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import get_dataloaders
from model import DesertSegFormer, CombinedLoss, get_optimizer_and_scheduler

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = "./data"  # Where your images are
CHECKPOINT_DIR = "./outputs/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16      # Use 16 on Colab T4. Use 4 or 8 on RTX 2050.
EPOCHS = 20
NUM_WORKERS = 2

def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, scaler):
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc="Training")

    for batch in loop:
        images = batch['image'].to(DEVICE)
        masks  = batch['mask'].to(DEVICE)

        # Mixed Precision Context (Speeds up training significantly)
        with torch.cuda.amp.autocast():
            logits = model(images)
            loss, ce, dice = loss_fn(logits, masks)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)

def validate(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(DEVICE)
            masks  = batch['mask'].to(DEVICE)
            
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss, _, _ = loss_fn(logits, masks)
            
            total_loss += loss.item()
    
    return total_loss / len(loader)

def main():
    print(f"Training on {DEVICE}...")
    
    # 1. Load Data
    train_loader, val_loader = get_dataloaders(DATA_DIR, BATCH_SIZE, NUM_WORKERS)
    
    # 2. Setup Model
    model = DesertSegFormer(variant='b2', num_classes=6, pretrained=True).to(DEVICE)
    
    # 3. Setup Optimizer & Loss
    optimizer, scheduler = get_optimizer_and_scheduler(model, epochs=EPOCHS, steps_per_epoch=len(train_loader))
    loss_fn = CombinedLoss()
    scaler = torch.cuda.amp.GradScaler() # For FP16 training

    best_val_loss = float('inf')

    # 4. Training Loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, scaler)
        val_loss   = validate(model, val_loader, loss_fn)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f"{CHECKPOINT_DIR}/best_model.pth"
            model.save_model(save_path)
            print(f"Saved Best Model to {save_path}")

        # Save Last Model (Safety)
        model.save_model(f"{CHECKPOINT_DIR}/last.pth")

if __name__ == "__main__":
    main()