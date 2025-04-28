import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskNet(nn.Module):
    def __init__(self, input_dim, task_output_dims):
        """
        input_dim: number of input features
        task_output_dims: dictionary having info for each data type
        """

        super(MultiTaskNet, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.Dropout(0.3)
        )

        # added batch normalization
        # self.shared = nn.Sequential(
        #     nn.Linear(input_dim, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(1024, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU()
        # )


        self.heads = nn.ModuleDict()
        for task, out_dim in task_output_dims.items():
            self.heads[task] = nn.Linear(256, out_dim)
            
            # tried deep head
            # self.heads[task] = nn.Sequential(
            #     nn.Linear(256, 128),
            #     nn.ReLU(),
            #     nn.Dropout(0.2),
            #     nn.Linear(128, out_dim)
            # )

            # Task-specific output heads
            # self.heads = nn.ModuleDict({
            #     task: nn.Linear(128, out_dim)
            #     for task, out_dim in task_output_dims.items()
            # })


    def forward(self, x , task):
        shared_out = self.shared(x)
        out = self.heads[task](shared_out)
        return out


# MTL refers to training a neural network to perform multiple tasks by sharing some of the network’s layers and parameters across tasks.
# In MTL, the goal is to improve the generalization performance of the model by leveraging the information shared across tasks
# The most common approach is to use a shared feature extractor and multiple task-specific heads.