import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

def save_results(results, save_path, config=None):
    """Save training results to disk"""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Save results as pickle
    with open(f"{save_path}/results.pkl", "wb") as f:
        pickle.dump(results, f)
    
    # Save config if provided
    if config is not None:
        import yaml
        with open(f"{save_path}/config.yaml", "w") as f:
            yaml.dump(config, f)
    
    print(f"Results saved to {save_path}")

def load_results(load_path):
    """Load training results from disk"""
    with open(f"{load_path}/results.pkl", "rb") as f:
        results = pickle.load(f)
    return results

def plot_training_curves(all_results, save_path, agent_name):
    """
    Plot aggregated training curves from multiple seeds
    
    Args:
        all_results: List of results dictionaries from different seeds
        save_path: Path to save plots
        agent_name: Name of agent (e.g., "agent0")
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Extract data from all seeds
    num_seeds = len(all_results)
    
    # 1. Plot training episode returns
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode returns during training
    ax = axes[0, 0]
    for i, results in enumerate(all_results):
        if results['episode_returns']:
            steps = [r['step'] for r in results['episode_returns']]
            returns = [r['return'] for r in results['episode_returns']]
            ax.plot(steps, returns, alpha=0.3, label=f'Seed {i}')
    
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Episode Return')
    ax.set_title('Training Episode Returns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Evaluation returns (aggregated)
    ax = axes[0, 1]
    eval_steps = all_results[0]['eval_steps'] if all_results[0]['eval_steps'] else []
    
    if eval_steps:
        eval_returns_all = np.array([r['eval_returns'] for r in all_results])
        mean_eval = np.mean(eval_returns_all, axis=0)
        min_eval = np.min(eval_returns_all, axis=0)
        max_eval = np.max(eval_returns_all, axis=0)
        
        ax.plot(eval_steps, mean_eval, 'b-', linewidth=2, label='Mean')
        ax.fill_between(eval_steps, min_eval, max_eval, alpha=0.3, label='Min/Max')
        ax.set_xlabel('Environment Steps')
        ax.set_ylabel('Evaluation Return')
        ax.set_title('Evaluation Returns (Greedy Policy)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=500, color='r', linestyle='--', label='Optimal', alpha=0.5)
    
    # Actor loss
    ax = axes[1, 0]
    for i, results in enumerate(all_results):
        if results['actor_losses']:
            steps = [l['step'] for l in results['actor_losses']]
            losses = [l['loss'] for l in results['actor_losses']]
            ax.plot(steps, losses, alpha=0.5, label=f'Seed {i}')
    
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Actor Loss')
    ax.set_title('Actor Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Critic loss
    ax = axes[1, 1]
    for i, results in enumerate(all_results):
        if results['critic_losses']:
            steps = [l['step'] for l in results['critic_losses']]
            losses = [l['loss'] for l in results['critic_losses']]
            ax.plot(steps, losses, alpha=0.5, label=f'Seed {i}')
    
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Critic Loss')
    ax.set_title('Critic Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{agent_name.upper()} - Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_curves.png', dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {save_path}/training_curves.png")
    plt.close()
    
    # 2. Plot value functions (if available)
    if all_results[0]['value_functions'] and any(all_results[0]['value_functions']):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for eval_idx, step in enumerate(eval_steps):
            for seed_idx, results in enumerate(all_results):
                if eval_idx < len(results['value_functions']):
                    values = results['value_functions'][eval_idx]
                    if values:
                        ax.plot(values, alpha=0.3, label=f'Step {step}, Seed {seed_idx}')
        
        ax.set_xlabel('Timestep in Episode')
        ax.set_ylabel('Value Function Estimate')
        ax.set_title('Value Function Evolution')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}/value_functions.png', dpi=150, bbox_inches='tight')
        print(f"Value functions saved to {save_path}/value_functions.png")
        plt.close()

def plot_comparison(agents_results, agent_names, save_path):
    """
    Compare multiple agents on the same plot
    
    Args:
        agents_results: List of lists of results (one list per agent, containing results from multiple seeds)
        agent_names: List of agent names
        save_path: Path to save comparison plot
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for agent_results, agent_name in zip(agents_results, agent_names):
        eval_steps = agent_results[0]['eval_steps']
        eval_returns_all = np.array([r['eval_returns'] for r in agent_results])
        mean_eval = np.mean(eval_returns_all, axis=0)
        std_eval = np.std(eval_returns_all, axis=0)
        
        ax.plot(eval_steps, mean_eval, linewidth=2, label=agent_name)
        ax.fill_between(eval_steps, 
                        mean_eval - std_eval, 
                        mean_eval + std_eval, 
                        alpha=0.2)
    
    ax.axhline(y=500, color='r', linestyle='--', label='Optimal', alpha=0.5)
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Evaluation Return')
    ax.set_title('Agent Comparison - Evaluation Returns')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/agent_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}/agent_comparison.png")
    plt.close()