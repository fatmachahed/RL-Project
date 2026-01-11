# train.py
import torch
import torch.optim as optim
import gymnasium as gym
import yaml
import argparse
from a2c_agent import A2CAgent
from model import ActorCritic
from utils.plotting import plot_training_curves

def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/agent0.yaml",
                        help="Path to the config file")
    args = parser.parse_args()

    config = load_config(args.config)

    # Conversion des hyperparamètres
    lr_actor = float(config["lr_actor"])
    lr_critic = float(config["lr_critic"])
    gamma = float(config["gamma"])
    n_steps = int(config["n_steps"])
    stochastic_rewards = bool(config["stochastic_rewards"])

    # Créer l'environnement
    env = gym.make("CartPole-v1", render_mode="human")  # "human" pour visualiser

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Créer le réseau Actor-Critic
    model = ActorCritic(state_dim, action_dim, hidden_size=64)

    # Optimizers
    optimizer_actor = optim.Adam(model.actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(model.critic.parameters(), lr=lr_critic)

    # Créer l'agent
    agent = A2CAgent(
        model,
        optimizer_actor,
        optimizer_critic,
        gamma=gamma,
        n_steps=n_steps,
        stochastic_rewards=stochastic_rewards
    )

    # Listes pour tracer les courbes
    all_rewards = []
    all_actor_losses = []
    all_critic_losses = []

    # max_steps = 500_000
    max_steps = 500
    step_counter = 0
    episode = 0

    while step_counter < max_steps:
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False
        episode_reward = 0

        while not done and step_counter < max_steps:
            action, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Stochastic reward
            if stochastic_rewards and torch.rand(1).item() < 0.9:
                reward = 0.0

            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

            # Stocker la transition
            agent.store_transition(state, action, reward, next_state_tensor, done)

            # Update
            actor_loss, critic_loss = agent.update()  # update doit maintenant renvoyer les losses
            all_actor_losses.append(actor_loss)
            all_critic_losses.append(critic_loss)

            state = next_state_tensor
            episode_reward += reward
            step_counter += 1

        episode += 1
        all_rewards.append(episode_reward)

        if episode % 10 == 0:
            print(f"Episode {episode}: reward = {episode_reward}")

    # Sauvegarder le modèle
    torch.save(model.state_dict(), "trained_models/trained_agent0.pth")
    print("Entraînement terminé, modèle sauvegardé sous 'trained_agent0.pth'")

    # Afficher les courbes
    plot_training_curves(all_rewards, all_actor_losses, all_critic_losses, label="Agent0")

if __name__ == "__main__":
    main()
