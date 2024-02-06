
import torch.nn as nn
import torch.nn.functional as F
import torch

class eeg_model(nn.Module):

    def __init__(self):
        super(eeg_model, self).__init__()
        num_inputs = 25
        num_outputs = 2
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, num_outputs),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)

