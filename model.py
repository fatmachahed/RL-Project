# models.py
import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.out(x)  # logits

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.out(x)  # value

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super().__init__()
        self.actor = Actor(state_dim, action_dim, hidden_size)
        self.critic = Critic(state_dim, hidden_size)

    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value
