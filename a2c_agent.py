import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import Actor, Critic

class A2CAgent:
    """Advantage Actor-Critic Agent"""
    
    def __init__(self, state_dim, action_dim, lr_actor=1e-5, lr_critic=1e-3, 
                 gamma=0.99, hidden_size=64):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            lr_actor: Learning rate for actor
            lr_critic: Learning rate for critic
            gamma: Discount factor
            hidden_size: Size of hidden layers (default 64)
        """
        self.gamma = gamma
        self.action_dim = action_dim
        
        # Create actor and critic networks
        self.actor = Actor(state_dim, action_dim, hidden_size)
        self.critic = Critic(state_dim, hidden_size)
        
        # Separate optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
    
    def select_action(self, state):
        """
        Select action given state(s)
        
        Args:
            state: Tensor of shape (batch_size, state_dim)
        
        Returns:
            actions: Tensor of shape (batch_size,)
            log_probs: Tensor of shape (batch_size,)
            values: Tensor of shape (batch_size,)
        """
        # Get action logits and value
        logits = self.actor(state)
        value = self.critic(state).squeeze(-1)
        
        # Sample action from categorical distribution
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value
    
    def select_action_greedy(self, state):
        """
        Select greedy action (for evaluation)
        
        Args:
            state: Tensor of shape (state_dim,) or (1, state_dim)
        
        Returns:
            action: Integer action
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            logits = self.actor(state)
            action = torch.argmax(logits, dim=-1)
        
        return action.item()
    
    def update(self, batch):
        """
        Update actor and critic networks
        
        Args:
            batch: Dictionary containing:
                - states: (n_steps, num_workers, state_dim)
                - actions: (n_steps, num_workers)
                - returns: (n_steps, num_workers) - n-step returns
                - values: (n_steps, num_workers) - old value estimates
        
        Returns:
            actor_loss: Float
            critic_loss: Float
        """
        # Extract batch data
        states = torch.FloatTensor(batch['states'])
        actions = torch.LongTensor(batch['actions'])
        returns = torch.FloatTensor(batch['returns'])
        old_values = torch.FloatTensor(batch['values'])
        
        # Flatten batch dimensions (n_steps, num_workers) -> (n_steps * num_workers)
        n_steps, num_workers, state_dim = states.shape
        states = states.reshape(-1, state_dim)
        actions = actions.reshape(-1)
        returns = returns.reshape(-1)
        old_values = old_values.reshape(-1)
        
        # Forward pass
        logits = self.actor(states)
        values = self.critic(states).squeeze(-1)
        
        # Compute advantages
        advantages = returns - values.detach()
        
        # Actor loss (policy gradient with advantage)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        
        actor_loss = -(log_probs * advantages).mean()
        
        # Critic loss (MSE between predicted value and return)
        critic_loss = F.mse_loss(values, returns)
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
    
    def save(self, path):
        """Save model weights"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model weights"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])