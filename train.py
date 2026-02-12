# train.py
import torch
import gymnasium as gym
import yaml
import argparse
import numpy as np
from pathlib import Path

from a2c_agent import A2CAgent
from model import ActorCritic
from rollout import Rollout
from evaluate import evaluate
from utils.plotting import plot_training_curves, save_results

def load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

def create_env(stochastic_rewards=False, seed=None):
    """Factory function to create a single environment"""
    env = gym.make("CartPole-v1")  # NO render_mode here!
    if seed is not None:
        env.reset(seed=seed)
    
    # Wrap for stochastic rewards if needed (Agent 1)
    if stochastic_rewards:
        from utils.env_wrappers import StochasticRewardWrapper
        env = StochasticRewardWrapper(env, mask_prob=0.9)
    
    return env

def train_single_run(config, seed):
    """Train one agent with given seed"""
    print(f"\n{'='*50}")
    print(f"Training with seed {seed}")
    print(f"K={config['K']}, n={config['n_steps']}")
    print(f"{'='*50}\n")
    
    # Create environment factory
    env_fn = lambda: create_env(
        stochastic_rewards=config['stochastic_rewards'],
        seed=seed
    )
    
    # Create rollout manager (handles K workers, n-step collection)
    rollout = Rollout(
        env_fn=env_fn,
        num_workers=config['K'],
        n_steps=config['n_steps']
    )
    
    # Create agent
    state_dim = 4  # CartPole state dimension
    action_dim = 2  # CartPole actions (left, right)
    
    agent = A2CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=float(config['lr_actor']),
        lr_critic=float(config['lr_critic']),
        gamma=float(config['gamma'])
    )
    
    # Training metrics
    results = {
        'episode_returns': [],
        'actor_losses': [],
        'critic_losses': [],
        'eval_returns': [],
        'eval_steps': [],
        'value_functions': []
    }
    
    # Training loop
    total_steps = 0
    max_steps = 500_000
    episode_rewards = []
    episode_steps = 0
    
    while total_steps < max_steps:
        # Collect K*n samples
        batch = rollout.collect_rollout(agent, gamma=config['gamma'])
        
        # Update agent
        actor_loss, critic_loss = agent.update(batch)
        
        # Log losses
        results['actor_losses'].append({
            'step': total_steps,
            'loss': actor_loss
        })
        results['critic_losses'].append({
            'step': total_steps,
            'loss': critic_loss
        })
        
        # Track episode returns from rollout
        if hasattr(rollout, 'episode_returns') and rollout.episode_returns:
            for ret in rollout.episode_returns:
                results['episode_returns'].append({
                    'step': total_steps,
                    'return': ret
                })
            rollout.episode_returns = []  # Clear
        
        total_steps += config['K'] * config['n_steps']
        
        # Evaluate every 20k steps
        if total_steps % 20_000 == 0:
            print(f"Step {total_steps}/{max_steps} - Evaluating...")
            
            # Create fresh evaluation environment
            eval_env = create_env(stochastic_rewards=False, seed=seed+10000)
            eval_results = evaluate(agent, eval_env, num_episodes=10)
            
            results['eval_returns'].append(eval_results['mean_return'])
            results['eval_steps'].append(total_steps)
            results['value_functions'].append(eval_results.get('value_trajectory', []))
            
            print(f"  Eval Return: {eval_results['mean_return']:.2f}")
            eval_env.close()
        
        # Log progress every 10k steps
        if total_steps % 10_000 == 0:
            recent_returns = [r['return'] for r in results['episode_returns'][-10:]]
            if recent_returns:
                print(f"Step {total_steps}: Avg Return (last 10 eps) = {np.mean(recent_returns):.2f}")
    
    # Close environments
    rollout.close()
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/agent0.yaml",
                        help="Path to the config file")
    parser.add_argument("--seeds", nargs='+', type=int, default=[0, 1, 2],
                        help="Random seeds for multiple runs")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Extract agent name from config path (e.g., "agent0" from "configs/agent0.yaml")
    agent_name = Path(args.config).stem
    
    # Run training with multiple seeds
    all_results = []
    for seed in args.seeds:
        results = train_single_run(config, seed)
        all_results.append(results)
        
        # Save individual run
        save_path = f"experiments/{agent_name}/seed_{seed}"
        save_results(results, save_path, config)
    
    # Plot aggregated results (mean + min/max shaded)
    plot_training_curves(
        all_results, 
        save_path=f"experiments/{agent_name}/plots",
        agent_name=agent_name
    )
    
    print(f"\n{'='*50}")
    print(f"Training complete! Results saved to experiments/{agent_name}/")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()