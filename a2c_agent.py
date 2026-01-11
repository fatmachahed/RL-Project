# a2c_agent.py
import torch
import torch.nn.functional as F

class A2CAgent:
    def __init__(self, model, optimizer_actor, optimizer_critic, gamma=0.99, n_steps=1, stochastic_rewards=False):
        self.model = model
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        self.gamma = gamma
        self.n_steps = n_steps
        self.stochastic_rewards = stochastic_rewards

        # stocker les transitions
        self.log_probs = []
        self.values = []
        self.rewards = []

    def select_action(self, state):
        logits, value = self.model(state)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1).item()  # int
        log_prob = torch.log(probs[0, action])

        # Stocker log_prob et value pour update
        self.log_probs.append(log_prob)
        self.values.append(value)

        return action, value

    def store_transition(self, state, action, reward, next_state, done):
        self.rewards.append(reward)

    def compute_loss(self):
        # calcul des retours
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.cat(self.values).squeeze()
        log_probs = torch.stack(self.log_probs)

        advantage = returns - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        return actor_loss, critic_loss

    def update(self):
        if len(self.rewards) == 0:
            return 0, 0
        actor_loss, critic_loss = self.compute_loss()

        # mise à jour des paramètres
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # reset
        self.rewards = []
        self.log_probs = []
        self.values = []

        return actor_loss.item(), critic_loss.item()
