import os
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(".")) 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from utils import get_dataloader
from models.mtl_model import MultiTaskNet

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

task_to_classes = {
    "pbmc68k": 11,
}

# Load sample to get input dim
sample_loader = get_dataloader("pbmc68k", batch_size=1, split="train")
input_dim = next(iter(sample_loader))[0].shape[1]

# Initialize model, optimizer, scheduler
model = MultiTaskNet(input_dim=input_dim, task_output_dims=task_to_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5, verbose=True)

# Training params
epochs = 10
batch_size = 64
early_stopping_patience = 5
save_path = "/pscratch/lji226_uksr/DMNN/results/PBMC_improved_multitask_model.pt"

# Track best score
best_val_f1 = defaultdict(float)
early_stopping_counter = defaultdict(int)

# === Training loop ===
for epoch in range(epochs):
    print(f"\n🌿 Epoch {epoch + 1}/{epochs}")

    for task in task_to_classes:
        num_classes = task_to_classes[task]

        train_loader = get_dataloader(task, batch_size=batch_size, split="train")
        val_loader   = get_dataloader(task, batch_size=batch_size, split="val", shuffle=False)

        # === Compute class weights ===
        all_targets = torch.cat([y for _, y in train_loader], dim=0).numpy()
        class_counts = torch.bincount(torch.tensor(all_targets), minlength=num_classes).float()
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = (class_weights / class_weights.sum()).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # === Train ===
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X, task)
            loss = criterion(outputs, y)
            loss.backward()

            # Optional: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        print(f" ✅ {task.upper()} Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

        # === Validate ===
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X, task)
                loss = criterion(outputs, y)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)

                correct += (preds == y).sum().item()
                total += y.size(0)

                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_acc = 100 * correct / total
        val_loss /= len(val_loader)
        val_f1 = f1_score(y_true, y_pred, average='macro')

        print(f" 📉 {task.upper()} Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%, F1: {val_f1:.4f}")

        # === LR Scheduling ===
        scheduler.step(val_f1)

        # === Save best model ===
        if val_f1 > best_val_f1[task]:
            print(f" 💾 New best F1 for {task.upper()}: {val_f1:.4f} — saving model...")
            best_val_f1[task] = val_f1
            torch.save(model.state_dict(), save_path)
            early_stopping_counter[task] = 0
        else:
            early_stopping_counter[task] += 1
            if early_stopping_counter[task] >= early_stopping_patience:
                print(f" ⏹️ Early stopping triggered for task {task.upper()}")

print("\n✅ Training completed.")
