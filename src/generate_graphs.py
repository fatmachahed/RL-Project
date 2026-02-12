import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import pandas as pd

# Configuration
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def create_plots_directory(project_root: Path):
    """Crée le dossier plots/ à la racine du projet"""
    plots_dir = project_root / "plots"
    plots_dir.mkdir(exist_ok=True)
    return plots_dir

def load_all_data(results_dir="results"):
    """Charge toutes les données de tous les agents et seeds"""
    data = {}
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ Dossier '{results_dir}' introuvable!")
        return data
    
    for agent_dir in results_path.iterdir():
        if agent_dir.is_dir():
            agent_name = agent_dir.name
            data[agent_name] = []
            
            for npz_file in agent_dir.glob("seed*.npz"):
                try:
                    run_data = np.load(npz_file, allow_pickle=True)
                    data[agent_name].append(dict(run_data))
                except Exception as e:
                    print(f"⚠️  Erreur chargement {npz_file}: {e}")
    
    return data


def get_config_info(run):
    """Extrait K et n_steps de la config"""
    if 'config' in run:
        config = run['config'].item() if hasattr(run['config'], 'item') else run['config']
        K = config.get('K', '?')
        n = config.get('n_steps', '?')
        stoch = config.get('stochastic_rewards', False)
        return K, n, stoch
    return '?', '?', False


# =========================================================
# 1. LEARNING CURVES (Eval Rewards)
# =========================================================
def plot_learning_curves(data, plots_dir):
    """Section 2.3: Evolution of average undiscounted returns across evaluations"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    agent_names = sorted(data.keys())
    
    for idx, agent in enumerate(agent_names):
        ax = axes[idx]
        
        # Plot individuel de chaque seed
        for run in data[agent]:
            if 'eval_steps' in run and 'eval_rewards' in run:
                ax.plot(run['eval_steps'], run['eval_rewards'], 
                       alpha=0.3, linewidth=1, color='gray')
        
        # Moyenne sur tous les seeds
        all_eval_steps = []
        all_eval_rewards = []
        
        for run in data[agent]:
            if 'eval_steps' in run and 'eval_rewards' in run:
                all_eval_steps.append(run['eval_steps'])
                all_eval_rewards.append(run['eval_rewards'])
        
        if all_eval_steps:
            max_steps = max(steps[-1] if len(steps) > 0 else 0 for steps in all_eval_steps)
            if max_steps > 0:
                common_steps = np.linspace(0, max_steps, 100)
                interpolated_rewards = []
                
                for steps, rewards in zip(all_eval_steps, all_eval_rewards):
                    if len(steps) > 0 and len(rewards) > 0:
                        interp_rewards = np.interp(common_steps, steps, rewards)
                        interpolated_rewards.append(interp_rewards)
                
                if interpolated_rewards:
                    mean_rewards = np.mean(interpolated_rewards, axis=0)
                    std_rewards = np.std(interpolated_rewards, axis=0)
                    
                    ax.plot(common_steps, mean_rewards, linewidth=2.5, 
                           label='Mean', color='darkblue')
                    ax.fill_between(common_steps, 
                                   mean_rewards - std_rewards,
                                   mean_rewards + std_rewards,
                                   alpha=0.2, color='darkblue', label='±1 std')
        
        K, n, stoch = get_config_info(data[agent][0]) if data[agent] else ('?', '?', False)
        title = f'{agent}\nK={K}, n={n}'
        if stoch:
            title += ', stoch=True'
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Eval Reward')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    for i in range(len(agent_names), 6):
        fig.delaxes(axes[i])
    
    plt.suptitle('Learning Curves - Evaluation Rewards', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = plots_dir / '1_learning_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvegardé: {output_path}")


# =========================================================
# 2. TRAINING REWARDS
# =========================================================
def plot_training_rewards(data, plots_dir):
    """Evolution of training rewards"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    agent_names = sorted(data.keys())
    
    for idx, agent in enumerate(agent_names):
        ax = axes[idx]
        
        all_steps = []
        all_means = []
        
        for run in data[agent]:
            if 'train_steps' in run and 'train_reward_mean' in run:
                all_steps.append(run['train_steps'])
                all_means.append(run['train_reward_mean'])
        
        if all_steps:
            max_len = max(len(s) for s in all_steps)
            padded_means = [np.pad(m, (0, max_len - len(m)), constant_values=np.nan) 
                           for m in all_means]
            
            mean_of_means = np.nanmean(padded_means, axis=0)
            std_of_means = np.nanstd(padded_means, axis=0)
            
            steps = all_steps[0] if len(all_steps[0]) == max_len else np.arange(max_len) * 1000
            
            ax.plot(steps[:len(mean_of_means)], mean_of_means, 
                   linewidth=2, color='darkgreen', label='Mean')
            ax.fill_between(steps[:len(mean_of_means)], 
                           mean_of_means - std_of_means,
                           mean_of_means + std_of_means,
                           alpha=0.2, color='darkgreen', label='±1 std')
        
        K, n, stoch = get_config_info(data[agent][0]) if data[agent] else ('?', '?', False)
        title = f'{agent}\nK={K}, n={n}'
        if stoch:
            title += ', stoch=True'
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Train Reward')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    for i in range(len(agent_names), 6):
        fig.delaxes(axes[i])
    
    plt.suptitle('Training Rewards (logged each 1k steps)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = plots_dir / '2_training_rewards.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvegardé: {output_path}")


# =========================================================
# 3. LOSS DYNAMICS
# =========================================================
def plot_loss_dynamics(data, plots_dir):
    """Evolution of actor and critic losses"""
    agent_names = sorted(data.keys())
    n_agents = len(agent_names)
    
    fig, axes = plt.subplots(n_agents, 2, figsize=(15, 4*n_agents))
    
    if n_agents == 1:
        axes = axes.reshape(1, -1)
    
    for idx, agent in enumerate(agent_names):
        # Actor Loss
        ax_actor = axes[idx, 0]
        for run in data[agent]:
            if 'actor_losses' in run and 'train_steps' in run:
                steps = run['train_steps'][:len(run['actor_losses'])]
                ax_actor.plot(steps, run['actor_losses'], alpha=0.5)
        
        K, n, _ = get_config_info(data[agent][0]) if data[agent] else ('?', '?', False)
        ax_actor.set_title(f'{agent} - Actor Loss (K={K}, n={n})', fontweight='bold')
        ax_actor.set_xlabel('Steps')
        ax_actor.set_ylabel('Loss')
        ax_actor.grid(True, alpha=0.3)
        
        # Critic Loss
        ax_critic = axes[idx, 1]
        for run in data[agent]:
            if 'critic_losses' in run and 'train_steps' in run:
                steps = run['train_steps'][:len(run['critic_losses'])]
                ax_critic.plot(steps, run['critic_losses'], alpha=0.5)
        
        ax_critic.set_title(f'{agent} - Critic Loss (K={K}, n={n})', fontweight='bold')
        ax_critic.set_xlabel('Steps')
        ax_critic.set_ylabel('Loss')
        ax_critic.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = plots_dir / '3_loss_dynamics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvegardé: {output_path}")


# =========================================================
# 4. VALUE EVOLUTION (All + Zoom sans Agent0)
# =========================================================
def plot_value_evolution(data, plots_dir):
    """Value function evolution"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    
    agent_names = sorted(data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(agent_names)))
    
    # GAUCHE: Tous les agents
    ax_all = axes[0]
    for idx, agent in enumerate(agent_names):
        all_values = []
        for run in data[agent]:
            if 'value_means' in run:
                all_values.append(run['value_means'])
        
        if all_values:
            max_len = max(len(v) for v in all_values)
            padded = [np.pad(v, (0, max_len - len(v)), constant_values=np.nan) 
                     for v in all_values]
            
            mean_values = np.nanmean(padded, axis=0)
            std_values = np.nanstd(padded, axis=0)
            steps = np.arange(len(mean_values)) * 1000
            
            K, n, _ = get_config_info(data[agent][0])
            label = f'{agent} (K={K}, n={n})'
            
            ax_all.plot(steps, mean_values, color=colors[idx], linewidth=2, label=label)
            ax_all.fill_between(steps, mean_values - std_values, mean_values + std_values,
                               alpha=0.2, color=colors[idx])
    
    ax_all.set_title('Value Evolution – All Agents', fontsize=14, fontweight='bold')
    ax_all.set_xlabel('Steps')
    ax_all.set_ylabel('Mean Value Estimate')
    ax_all.legend()
    ax_all.grid(True, alpha=0.3)
    
    # DROITE: Sans Agent0 (zoom)
    ax_zoom = axes[1]
    for idx, agent in enumerate(agent_names):
        if 'agent0' in agent.lower():
            continue
        
        all_values = []
        for run in data[agent]:
            if 'value_means' in run:
                all_values.append(run['value_means'])
        
        if all_values:
            max_len = max(len(v) for v in all_values)
            padded = [np.pad(v, (0, max_len - len(v)), constant_values=np.nan) 
                     for v in all_values]
            
            mean_values = np.nanmean(padded, axis=0)
            std_values = np.nanstd(padded, axis=0)
            steps = np.arange(len(mean_values)) * 1000
            
            K, n, _ = get_config_info(data[agent][0])
            label = f'{agent} (K={K}, n={n})'
            
            ax_zoom.plot(steps, mean_values, linewidth=2, label=label)
            ax_zoom.fill_between(steps, mean_values - std_values, mean_values + std_values,
                                alpha=0.2)
    
    ax_zoom.set_title('Value Evolution – Without Agent0', fontsize=14, fontweight='bold')
    ax_zoom.set_xlabel('Steps')
    ax_zoom.set_ylabel('Mean Value Estimate')
    ax_zoom.legend()
    ax_zoom.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = plots_dir / '4_value_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvegardé: {output_path}")


# =========================================================
# 5. TRAJECTORY VALUES
# =========================================================
def plot_trajectory_values(data, plots_dir):
    """Value function on full trajectories"""
    agent_names = sorted(data.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, agent in enumerate(agent_names):
        ax = axes[idx]
        
        trajectories = []
        for run in data[agent]:
            if 'trajectories' in run:
                trajs = run['trajectories']
                if isinstance(trajs, np.ndarray) and len(trajs) > 0:
                    trajectories.extend(trajs)
        
        if trajectories:
            last_traj = trajectories[-1]
            if isinstance(last_traj, dict) and 'data' in last_traj:
                traj_data = last_traj['data']
                if 'values' in traj_data:
                    values = traj_data['values']
                    timesteps = np.arange(len(values))
                    
                    ax.plot(timesteps, values, linewidth=2, color='purple')
                    ax.fill_between(timesteps, 0, values, alpha=0.2, color='purple')
                    
                    step = last_traj.get('step', '?')
                    K, n, _ = get_config_info(data[agent][0]) if data[agent] else ('?', '?', False)
                    ax.set_title(f'{agent} - Trajectory Values\n(K={K}, n={n}, step={step})', 
                               fontweight='bold')
                    ax.set_xlabel('Timestep')
                    ax.set_ylabel('V(s)')
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No values', ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'Invalid format', ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'No trajectory', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    for i in range(len(agent_names), 6):
        fig.delaxes(axes[i])
    
    plt.suptitle('Value Function on Trajectories', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = plots_dir / '5_trajectory_values.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvegardé: {output_path}")


# =========================================================
# 6. ENTROPY EVOLUTION (All + Zoom)
# =========================================================
def plot_entropy_evolution(data, plots_dir):
    """Entropy evolution"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    
    agent_names = sorted(data.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(agent_names)))
    
    # GAUCHE: All agents
    ax_all = axes[0]
    for idx, agent in enumerate(agent_names):
        all_entropies = []
        for run in data[agent]:
            if 'entropies' in run:
                all_entropies.append(run['entropies'])
        
        if all_entropies:
            max_len = max(len(e) for e in all_entropies)
            padded = [np.pad(e, (0, max_len - len(e)), constant_values=np.nan)
                     for e in all_entropies]
            
            mean_entropy = np.nanmean(padded, axis=0)
            std_entropy = np.nanstd(padded, axis=0)
            steps = np.arange(len(mean_entropy)) * 1000
            
            K, n, _ = get_config_info(data[agent][0])
            label = f'{agent} (K={K}, n={n})'
            
            ax_all.plot(steps, mean_entropy, color=colors[idx],
                       linewidth=2, label=label)
            ax_all.fill_between(steps, mean_entropy - std_entropy, mean_entropy + std_entropy,
                               alpha=0.2, color=colors[idx])
    
    ax_all.set_title('Entropy Evolution – All Agents', fontweight='bold')
    ax_all.set_xlabel('Steps')
    ax_all.set_ylabel('Entropy')
    ax_all.legend()
    ax_all.grid(True, alpha=0.3)
    
    # DROITE: Without Agent0
    ax_zoom = axes[1]
    for idx, agent in enumerate(agent_names):
        if 'agent0' in agent.lower():
            continue
        
        all_entropies = []
        for run in data[agent]:
            if 'entropies' in run:
                all_entropies.append(run['entropies'])
        
        if all_entropies:
            max_len = max(len(e) for e in all_entropies)
            padded = [np.pad(e, (0, max_len - len(e)), constant_values=np.nan)
                     for e in all_entropies]
            
            mean_entropy = np.nanmean(padded, axis=0)
            std_entropy = np.nanstd(padded, axis=0)
            steps = np.arange(len(mean_entropy)) * 1000
            
            K, n, _ = get_config_info(data[agent][0])
            label = f'{agent} (K={K}, n={n})'
            
            ax_zoom.plot(steps, mean_entropy, linewidth=2, label=label)
            ax_zoom.fill_between(steps, mean_entropy - std_entropy, mean_entropy + std_entropy,
                                alpha=0.2)
    
    ax_zoom.set_title('Entropy Evolution – Without Agent0', fontweight='bold')
    ax_zoom.set_xlabel('Steps')
    ax_zoom.set_ylabel('Entropy')
    ax_zoom.legend()
    ax_zoom.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = plots_dir / '6_entropy_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvegardé: {output_path}")



# =========================================================
# 7. PERFORMANCE COMPARISON
# =========================================================
def plot_performance_comparison(data, plots_dir):
    """Final performance boxplots"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    agent_names = sorted(data.keys())
    final_rewards = []
    final_lengths = []
    
    for agent in agent_names:
        rewards_per_agent = []
        lengths_per_agent = []
        
        for run in data[agent]:
            if 'eval_rewards' in run and len(run['eval_rewards']) > 0:
                last_n = min(5, len(run['eval_rewards']))
                rewards_per_agent.append(np.mean(run['eval_rewards'][-last_n:]))
            
            if 'eval_lengths' in run and len(run['eval_lengths']) > 0:
                last_n = min(5, len(run['eval_lengths']))
                lengths_per_agent.append(np.mean(run['eval_lengths'][-last_n:]))
        
        final_rewards.append(rewards_per_agent)
        final_lengths.append(lengths_per_agent)
    
    # Rewards
    ax1 = axes[0]
    bp1 = ax1.boxplot(final_rewards, labels=agent_names, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')
    
    ax1.set_title('Final Performance (mean of last 5 evals)', fontweight='bold')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=45)
    
    # Lengths
    ax2 = axes[1]
    bp2 = ax2.boxplot(final_lengths, labels=agent_names, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightcoral')
    
    ax2.set_title('Episode Lengths', fontweight='bold')
    ax2.set_ylabel('Length')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_path = plots_dir / '7_performance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvegardé: {output_path}")


# =========================================================
# 8. TRAINING STABILITY
# =========================================================
def plot_training_stability(data, plots_dir):
    """Training stability metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    agent_names = sorted(data.keys())
    
    # 1. Variance distribution
    ax1 = axes[0, 0]
    for agent in agent_names:
        eval_rewards_variance = []
        
        for run in data[agent]:
            if 'eval_rewards' in run:
                window = 5
                if len(run['eval_rewards']) >= window:
                    variances = [np.var(run['eval_rewards'][i:i+window]) 
                               for i in range(len(run['eval_rewards']) - window + 1)]
                    eval_rewards_variance.extend(variances)
        
        if eval_rewards_variance:
            ax1.hist(eval_rewards_variance, alpha=0.5, label=agent, bins=20)
    
    ax1.set_title('Reward Variance Distribution', fontweight='bold')
    ax1.set_xlabel('Variance')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Coefficient of variation
    ax2 = axes[0, 1]
    cvs = []
    
    for agent in agent_names:
        agent_cvs = []
        for run in data[agent]:
            if 'eval_rewards' in run and len(run['eval_rewards']) > 10:
                mean_r = np.mean(run['eval_rewards'])
                std_r = np.std(run['eval_rewards'])
                if mean_r != 0:
                    agent_cvs.append(std_r / abs(mean_r))
        cvs.append(agent_cvs)
    
    bp = ax2.boxplot(cvs, labels=agent_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
    
    ax2.set_title('Coefficient of Variation', fontweight='bold')
    ax2.set_ylabel('CV (std/mean)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Actor loss std
    ax3 = axes[1, 0]
    for agent in agent_names:
        actor_loss_std = []
        for run in data[agent]:
            if 'actor_losses' in run:
                actor_loss_std.append(np.std(run['actor_losses']))
        
        if actor_loss_std:
            ax3.scatter([agent]*len(actor_loss_std), actor_loss_std, 
                       alpha=0.6, s=100)
    
    ax3.set_title('Actor Loss Std', fontweight='bold')
    ax3.set_ylabel('Std(Actor Loss)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Improvement (2nd half - 1st half)
    ax4 = axes[1, 1]
    improvements = []
    
    for agent in agent_names:
        agent_improvements = []
        for run in data[agent]:
            if 'eval_rewards' in run and len(run['eval_rewards']) > 10:
                mid = len(run['eval_rewards']) // 2
                first_half = np.mean(run['eval_rewards'][:mid])
                second_half = np.mean(run['eval_rewards'][mid:])
                improvement = second_half - first_half
                agent_improvements.append(improvement)
        improvements.append(agent_improvements)
    
    bp2 = ax4.boxplot(improvements, labels=agent_names, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('lightyellow')
    
    ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax4.set_title('Improvement (2nd half - 1st half)', fontweight='bold')
    ax4.set_ylabel('Δ Reward')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = plots_dir / '8_training_stability.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvegardé: {output_path}")


# =========================================================
# 9. CORRELATION HEATMAP
# =========================================================
def plot_correlation_heatmap(data, plots_dir):
    """Correlation matrix between metrics"""
    n_agents = len(data)
    fig, axes = plt.subplots(1, n_agents, figsize=(6*n_agents, 5))
    
    if n_agents == 1:
        axes = [axes]
    
    for idx, agent in enumerate(sorted(data.keys())):
        metrics_dict = {
            'eval_rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'entropies': [],
            'value_means': []
        }
        
        for run in data[agent]:
            for key in metrics_dict.keys():
                if key in run:
                    values = run[key]
                    if hasattr(values, 'tolist'):
                        metrics_dict[key].extend(values.tolist())
                    elif isinstance(values, (list, np.ndarray)):
                        metrics_dict[key].extend(values)
        
        min_len = min(len(v) for v in metrics_dict.values() if len(v) > 0)
        
        if min_len > 0:
            df_data = {k: v[:min_len] for k, v in metrics_dict.items() if len(v) > 0}
            df = pd.DataFrame(df_data)
            corr = df.corr()
            
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, ax=axes[idx], 
                       cbar_kws={'label': 'Correlation'},
                       vmin=-1, vmax=1)
            
            K, n, _ = get_config_info(data[agent][0]) if data[agent] else ('?', '?', False)
            axes[idx].set_title(f'{agent} (K={K}, n={n})', fontweight='bold')
    
    plt.tight_layout()
    output_path = plots_dir / '9_correlation_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvegardé: {output_path}")


# =========================================================
# 10. CONVERGENCE SPEED (Steps vs Updates - CORRECT VERSION)
# =========================================================
def plot_convergence_speed(data, plots_dir, threshold_ratio=0.8):
    """Convergence speed: raw steps vs corrected updates"""
    agent_names = sorted(data.keys())
    convergence_steps = []
    convergence_updates = []
    convergence_data = []
    
    for agent in agent_names:
        agent_steps = []
        agent_updates = []
        
        # Extract K
        K = 1
        if data[agent]:
            K, _, _ = get_config_info(data[agent][0])
            K = int(K) if K != '?' else 1
        
        for run in data[agent]:
            if 'eval_rewards' not in run or 'eval_steps' not in run:
                continue
            
            rewards = run['eval_rewards']
            steps = run['eval_steps']
            
            if len(rewards) < 5:
                continue
            
            final_reward = np.mean(rewards[-5:])
            threshold = threshold_ratio * final_reward
            
            if final_reward < 50:  # CartPole minimum threshold
                continue
            
            # Find first step above threshold
            for r, s in zip(rewards, steps):
                if r >= threshold:
                    agent_steps.append(s)
                    agent_updates.append(s / K)  # ✅ CORRECT: divide by K
                    break
        
        convergence_steps.append(agent_steps)
        convergence_updates.append(agent_updates)
        
        if agent_updates:
            convergence_data.append({
                'agent': agent,
                'K': K,
                'mean_steps': np.mean(agent_steps),
                'std_steps': np.std(agent_steps),
                'mean_updates': np.mean(agent_updates),
                'std_updates': np.std(agent_updates)
            })
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # LEFT: Raw steps (biased for K>1)
    ax1 = axes[0]
    bp1 = ax1.boxplot(convergence_steps, labels=agent_names, patch_artist=True)
    
    colors_box = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightsteelblue']
    for patch, color in zip(bp1['boxes'], colors_box[:len(agent_names)]):
        patch.set_facecolor(color)
    
    ax1.set_title(f"Raw steps to {int(threshold_ratio*100)}% reward\n(Biased for K>1)",
                 fontsize=12, fontweight="bold")
    ax1.set_ylabel("Environment Steps")
    ax1.set_xlabel("Agent")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.tick_params(axis='x', rotation=45)
    
    # RIGHT: Updates (fair comparison) ✅
    ax2 = axes[1]
    
    if convergence_data:
        agents = [d['agent'] for d in convergence_data]
        means = [d['mean_updates'] for d in convergence_data]
        stds = [d['std_updates'] for d in convergence_data]
        Ks = [d['K'] for d in convergence_data]
        
        x_pos = np.arange(len(agents))
        
        # Colors based on K
        colors_bar = ['steelblue' if K == 1 else 'darkgreen' for K in Ks]
        
        bars = ax2.bar(x_pos, means, yerr=stds, capsize=5, 
                       color=colors_bar, alpha=0.7, edgecolor='black')
        
        # Annotations
        for i, (agent_name, bar, K) in enumerate(zip(agents, bars, Ks)):
            agent_data = data[agent_name][0] if agent_name in data and data[agent_name] else None
            if agent_data:
                _, n, _ = get_config_info(agent_data)
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'K={K}, n={n}\n{int(means[i])} updates',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(agents, rotation=45)
        ax2.set_ylabel('Updates (mean ± std)')
        ax2.set_title(f'Convergence Speed - Updates to {int(threshold_ratio*100)}% reward\n✅ Fair comparison',
                     fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = plots_dir / "10_convergence_speed.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Sauvegardé: {output_path}")
    
    # Console output
    print("\n📊 Convergence Speed Summary:")
    print("="*80)
    print(f"{'Agent':<10} {'K':>3} {'Steps (mean±std)':>25} {'Updates (mean±std)':>25}")
    print("="*80)
    for d in convergence_data:
        print(f"{d['agent']:<10} {d['K']:>3} "
              f"{d['mean_steps']:>8.0f} ± {d['std_steps']:>6.0f} steps   "
              f"{d['mean_updates']:>8.0f} ± {d['std_updates']:>6.0f} updates")
    print("="*80)
    print("💡 Updates = Steps / K (fair comparison)")


# =========================================================
# 11. FINAL REWARD PER SEED PER AGENT (échelle adaptative + titre)
# =========================================================
def plot_final_rewards_per_agent(data, plots_dir, seeds=[42, 123, 999]):
    """Affiche un graphe par agent avec les final rewards par seed,
       relie les barres, colorie le meilleur seed et adapte l'échelle par agent"""
    
    agent_names = sorted(data.keys())
    n_agents = len(agent_names)
    
    fig, axes = plt.subplots(1, n_agents, figsize=(5*n_agents, 5), sharey=False)
    
    if n_agents == 1:
        axes = [axes]
    
    for ax, agent in zip(axes, agent_names):
        final_rewards = []
        for seed in seeds:
            # Cherche le run correspondant au seed
            matched_run = None
            for run in data[agent]:
                if 'seed' in run and run['seed'] == seed:
                    matched_run = run
                    break
            # Sinon prend le premier run disponible
            if not matched_run:
                matched_run = data[agent][0]
            
            reward = matched_run['eval_rewards'][-1] if 'eval_rewards' in matched_run else 0
            final_rewards.append(reward)
        
        # Trouve la meilleure seed
        best_idx = np.argmax(final_rewards)
        
        # Dessine les barres
        colors = ['skyblue']*len(seeds)
        colors[best_idx] = 'steelblue'  # meilleure seed
        bars = ax.bar([str(s) for s in seeds], final_rewards, color=colors)
        
        # Relie les barres avec une ligne
        ax.plot([str(s) for s in seeds], final_rewards, color='gray', linestyle='--', marker='o')
        
        ax.set_title(f'{agent}', fontweight='bold')
        ax.set_xlabel('Seed')
        ax.set_ylabel('Final Eval Reward')
        ax.grid(axis='y', alpha=0.3)
        
        # Ajuste l'échelle automatiquement pour chaque agent
        ax.set_ylim(bottom=0, top=max(final_rewards)*1.2)
    
    # Titre général
    plt.suptitle('Final Evaluation Reward per Seed for Each Agent\n(Barre + ligne, meilleure seed en bleu foncé)', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    output_path = plots_dir / "11_final_rewards_per_agent.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvegardé: {output_path}")

