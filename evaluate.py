# evaluate.py
import gymnasium as gym
import torch
from model import ActorCritic

def evaluate(model, env_name="CartPole-v1", episodes=10, render=False):
    env = gym.make(env_name, render_mode="human" if render else None)
    total_rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                logits, _ = model(state_tensor)
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs).item()  # Greedy action
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            state = next_state

        total_rewards.append(ep_reward)
        print(f"Episode {ep+1}: Reward = {ep_reward}")

    avg_reward = sum(total_rewards) / episodes
    print(f"Average reward over {episodes} episodes: {avg_reward}")
    env.close()
    return total_rewards

if __name__ == "__main__":
    # Exemple : tester un modèle aléatoire
    state_dim = 4
    action_dim = 2
    model = ActorCritic(state_dim, action_dim)
    evaluate(model, episodes=5, render=True)
