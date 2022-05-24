import torch
import torch.nn as nn
import numpy as np

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_prob = F.softmax(self.action_head(x), dim=-1)
        return action_prob

    def sample_actions(x):
        probs = self.forward(x)
        m = Categorical(probs)
        action = m.sample()
        return action

class QCritic(nn.Module):
    def __init__(self):
        super(QCritic, self).__init__()
        self.affine1 = nn.Linear(6, 128)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x, u):
        x = torch.cat([x, u], dim=-1)
        x = F.relu(self.affine1(x))
        state_values = self.value_head(x)
        return state_values