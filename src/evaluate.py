import torch
import numpy as np
import gymnasium as gym
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# EVALUATION - Conforme Section 2.3
# =========================================================
def evaluate(agent, env_name, episodes=10, store_trajectory=False):
    """
    Section 2.3: "10 episodes with greedy action policy"
    Section 2.3: "Plot value function on one full trajectory"
    """
    env = gym.make(env_name)
    rewards = []
    lengths = []
    trajectory = None

    for ep_idx in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        
        # Stocker UNE trajectoire (le 1er épisode)
        if store_trajectory and ep_idx == 0:
            traj_states = []
            traj_values = []
            traj_actions = []
            traj_rewards = []

        while not done:
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, value = agent.model(s)
                torch.distributions.Categorical(logits=logits)
                action = torch.argmax(logits, dim=-1)  # Greedy policy (sampling from learned π)

            next_state, r, term, trunc, _ = env.step(action.item())
            done = term or trunc

            if store_trajectory and ep_idx == 0:
                traj_states.append(state.copy())
                traj_values.append(value.item())
                traj_actions.append(action.item())
                traj_rewards.append(r)

            ep_reward += r
            state = next_state

        rewards.append(ep_reward)
        lengths.append(len(traj_states) if (store_trajectory and ep_idx == 0) else 0)
        
        if store_trajectory and ep_idx == 0:
            trajectory = {
                'states': np.array(traj_states),
                'values': np.array(traj_values),
                'actions': np.array(traj_actions),
                'rewards': np.array(traj_rewards)
            }

    env.close()
    return np.mean(rewards), np.mean(lengths), trajectory

