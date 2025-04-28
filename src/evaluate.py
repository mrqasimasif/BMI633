import os
import sys
sys.path.append(os.path.abspath(".")) 

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from utils import get_dataloader
from models.mtl_model import MultiTaskNet


# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

task_to_classes = {
    # "cbmc": 13,
    # "goolam": 5,
    # "melanoma": 8,
    # "yan": 6,
    # "klein_mouse": 4,
    # "klein_human": 4,
    "pbmc68k": 11,
}


# Load one sample to get input dimension
sample_loader = get_dataloader("pbmc68k", batch_size=1, split="test")
input_dim = next(iter(sample_loader))[0].shape[1]

# input dim specifies / restricts the domain

# Load model
model = MultiTaskNet(input_dim=input_dim, task_output_dims=task_to_classes).to(device)
checkpoint_path = "/pscratch/lji226_uksr/DMNN/results/PBMC_improved_multitask_model.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Evaluation loop
for task in task_to_classes:
    print(f"\n🔍 Evaluating task: {task}")
    test_loader = get_dataloader(task, batch_size=128, split="test", shuffle=False)

    all_preds, all_labels = [], []
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            logits = model(X, task)
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    print(f"    ✅ Accuracy: {acc*100:.2f}% | F1 Score: {f1:.3f}")

    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds))

    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title(f"Confusion Matrix - {task}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    os.makedirs("/pscratch/lji226_uksr/DMNN/results/figures", exist_ok=True)
    plt.savefig(f"/pscratch/lji226_uksr/DMNN/results/figures/{task}_improved_confusion_matrix.png")
    plt.close()
