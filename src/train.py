import os
import sys
sys.path.append(os.path.abspath(".")) 

from collections import defaultdict

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
    #"cbmc": 13,
    #"goolam": 5,
    #"melanoma": 8,
    #"klein_mouse": 4,
    #"klein_human": 4,

    #"yan": 6,
    "pbmc68k": 11,
}

# Load one sample to get input dimension
sample_loader = get_dataloader("pbmc68k", batch_size=1, split="train")
input_dim = next(iter(sample_loader))[0].shape[1]


# Initialize model
model = MultiTaskNet(input_dim=input_dim, task_output_dims=task_to_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Training params
epochs = 10
batch_size = 64

# Track best score
best_val_f1 = defaultdict(float)
early_stopping_counter = defaultdict(int)


for epoch in range(epochs):
    print(f"\n🌿 Epoch {epoch + 1}/{epochs}")

    for task in task_to_classes:
        print(f"🔁 Training on task: {task}")

        train_loader = get_dataloader(task, batch_size=batch_size, split="train")
        val_loader   = get_dataloader(task, batch_size=batch_size, split="val", shuffle=False)

        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X, task)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        print(f" ✅ Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

        # 📊 Validation loop
        model.eval()
        with torch.no_grad():
            correct, total, val_loss = 0, 0, 0.0
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X, task)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            val_acc = 100 * correct / total
            val_loss /= len(val_loader)
            print(f" 📉 Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

# Save model
os.makedirs("/pscratch/lji226_uksr/DMNN/results", exist_ok=True)
torch.save(model.state_dict(), "/pscratch/lji226_uksr/DMNN/results/PBMCmultitask_model.pt")
print("✅ Model saved.")