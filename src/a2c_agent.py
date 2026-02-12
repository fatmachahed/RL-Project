import torch
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# A2C AGENT - Conforme Figure 2
# =========================================================
class A2CAgent:
    def __init__(self, model, optimizer, gamma=0.99, entropy_coef=0.01):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.reset_buffer()

    def reset_buffer(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.entropies = []

    def act(self, states):
        """Sélectionne actions selon politique stochastique"""
        logits, values = self.model(states)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        return actions, dist.log_prob(actions), values, dist.entropy()

    def compute_returns(self, next_value):
        """
        Calcule n-step returns avec bootstrapping correct
        Section 3.1: Bootstrap at truncation, NOT at termination
        """
        R = next_value  # Shape: [K]
        returns = []
        
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            # d=1 SEULEMENT si terminal (pole tombé)
            # d=0 si truncation (500 steps) → on bootstrap
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        
        return torch.cat(returns, dim=0)

    def learn(self, next_value):
        """
        Mise à jour A2C conforme Figure 2
        """
        returns = self.compute_returns(next_value)
        values = torch.cat(self.values, dim=0)
        log_probs = torch.cat(self.log_probs, dim=0)
        entropies = torch.cat(self.entropies, dim=0)
        
        assert returns.shape == values.shape
        
        # Advantages A^(k) = R^(k) - V(s^(k))
        advantages = returns - values.detach()
        
        # Normalisation pour stabilité
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        
        # Pertes (Figure 2)
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = 0.5 * (returns - values).pow(2).mean()
        entropy = entropies.mean()
        
        # Loss totale
        loss = actor_loss + critic_loss - self.entropy_coef * entropy

        # Optimisation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        self.reset_buffer()
        return actor_loss.item(), critic_loss.item(), entropy.item()

