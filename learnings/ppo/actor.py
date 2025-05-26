import torch.nn as nn
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x, action_mask):
        logits = self.model(x)

        if logits.dtype != action_mask.dtype:
            action_mask = action_mask.bool()

        invalid_mask = (~action_mask).float()
        masked_logits = logits - invalid_mask * 1e9

        return Categorical(logits=masked_logits)
