import os
import torch
from torch.utils.data import Dataset
import anndata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np


# no train test split
# class SingleCellDataset(Dataset):
#     def __init__(self, adata_path):
#             self.adata = anndata.read_h5ad(adata_path)
#             self.X = self.adata.X.toarray() if hasattr(self.adata.X, "toarray") else self.adata.X
#             self.y = LabelEncoder().fit_transform(self.adata.obs["cell_type"])
#             self.X = torch.tensor(self.X, dtype=torch.float32)
#             self.y = torch.tensor(self.y, dtype=torch.long)

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]



# included train test splot
class SingleCellDataset(Dataset):
    def __init__(self, adata_path, split="train", val_frac=0.1, test_frac=0.2, random_state=42):
        self.adata = anndata.read_h5ad(adata_path)
        X = self.adata.X.toarray() if hasattr(self.adata.X, "toarray") else self.adata.X
        y = LabelEncoder().fit_transform(self.adata.obs["cell_type"])
        
        # Split data
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_frac, stratify=y, random_state=random_state)
        if split == "test":
            self.X, self.y = X_test, y_test
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_frac, stratify=y_trainval, random_state=random_state)
            if split == "train":
                self.X, self.y = X_train, y_train
            elif split == "val":
                self.X, self.y = X_val, y_val
            else:
                raise ValueError(f"Invalid split: {split}")

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
