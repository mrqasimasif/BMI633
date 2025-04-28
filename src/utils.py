import os
from torch.utils.data import DataLoader
from dataset import SingleCellDataset

# def get_dataloader(task_name, batch_size=64, shuffle=True):
#     path = f"../data/processed/{task_name}.h5ad"
#     dataset = SingleCellDataset(path)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# fixed path
# def get_dataloader(task_name, batch_size=64, shuffle=True):
#     root = os.path.dirname(os.path.abspath(__file__))  # path to /src
#     path = os.path.join(root, "..", "data", "processed", f"{task_name}.h5ad")
#     dataset = SingleCellDataset(path)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# with train test split
def get_dataloader(task_name, batch_size=64, split="train", shuffle=True):
    root = os.path.dirname(os.path.abspath(__file__))  # path to /src
    path = os.path.join(root, "..", "data", "processed", f"{task_name}.h5ad")
    dataset = SingleCellDataset(path, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
