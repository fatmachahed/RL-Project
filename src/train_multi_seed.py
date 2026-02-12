from src.a2c_agent import A2CAgent
from src.model import ActorCritic
from src.evaluate import evaluate
import torch
import torch.optim as optim
import numpy as np
import gymnasium as gym
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# TRAINING 
# =========================================================
def train(config, seed, name):
    """
    Section 2.3: "500k environment steps"
    Section 2.3: "evaluate every 20k steps"
    Section 2.3: "log every 1k steps"
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Section 3.3: K parallel workers avec Gymnasium vectorized
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make(config["env_name"]) for _ in range(config["K"])]
    )

    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.n

    model = ActorCritic(obs_dim, act_dim).to(device)
    
    # Table 1: Adam with separate learning rates
    optimizer = optim.Adam([
        {"params": model.actor.parameters(), "lr": float(config["lr_actor"])},
        {"params": model.critic.parameters(), "lr": float(config["lr_critic"])}
    ])

    agent = A2CAgent(model, optimizer, config["gamma"], config["entropy_coef"])

    # =========================================================
    # MÉTRIQUES 
    # =========================================================
    metrics = {
        # Training metrics (logged each 1k steps)
        "train_steps": [],           # Steps où on log
        "train_reward_mean": [],     # Moyenne rewards épisodes terminés
        "train_reward_std": [],      # Écart-type
        "actor_losses": [],          # Actor loss
        "critic_losses": [],         # Critic loss (value loss)
        "entropies": [],             # Entropy
        "value_means": [],           # Mean value predictions
        "grad_norms": [],            # Gradient norms
        
        # Eval metrics (logged each 20k steps)
        "eval_steps": [],            # Steps d'évaluation
        "eval_rewards": [],          # Mean reward over 10 episodes
        "eval_lengths": [],          # Mean episode length
        
        # Trajectories (Section 2.3: "Plot value function on trajectory")
        "trajectories": []           # Stockage minimal
    }

    states, _ = envs.reset(seed=seed)
    states = torch.tensor(states, dtype=torch.float32).to(device)
    total_steps = 0
    last_log, last_eval = 0, 0
    
    # Episode tracking
    ep_rewards = np.zeros(config["K"])
    ep_steps = np.zeros(config["K"])
    
    # Buffers temporaires pour moyennage sur 1k steps
    recent_episode_rewards = []
    recent_values = []

    pbar = tqdm(total=config["max_steps"], desc=f"{name}-seed{seed}")

    while total_steps < config["max_steps"]:
        # Section 3.4: Collect n_steps before update
        for step_in_rollout in range(config["n_steps"]):
            actions, logp, values, entropy = agent.act(states)
            next_states, rewards, terms, truncs, _ = envs.step(actions.cpu().numpy())

            # =========================================================
            # Section 3.2: STOCHASTIC REWARDS
            # "mask such that reward is zeroed out with probability 0.9"
            # Clarification: agents 2, 3, 4 use reward masking
            # =========================================================
            learn_rewards = rewards.copy()
            if config.get("stochastic_rewards", False):
                # p=0.9 de zéroer → garde 10%
                mask = np.random.rand(*rewards.shape) < 0.1
                learn_rewards = rewards * mask

            # Stockage dans buffer
            agent.log_probs.append(logp)
            agent.values.append(values)
            agent.entropies.append(entropy)
            agent.rewards.append(
                torch.tensor(learn_rewards, dtype=torch.float32, device=device)
            )
            
            # Section 3.1: Correct bootstrapping
            # dones = 1.0 SEULEMENT si terminal (pole fell)
            # dones = 0.0 si truncation (500 steps reached) → on bootstrap
            dones = terms.astype(float)  # NE PAS inclure truncs!
            agent.dones.append(
                torch.tensor(dones, dtype=torch.float32, device=device)
            )

            # Section 3.2: "do not include masking in logging"
            # → Logger les VRAIES récompenses (sans masking)
            for i in range(config["K"]):
                ep_rewards[i] += rewards[i]  # Vraie récompense
                ep_steps[i] += 1
                
                if terms[i] or truncs[i]:
                    if ep_steps[i] > 0:
                        recent_episode_rewards.append(ep_rewards[i])
                    ep_rewards[i] = 0
                    ep_steps[i] = 0

            recent_values.append(values.detach().mean().item())
            states = torch.tensor(next_states, dtype=torch.float32).to(device)
            total_steps += config["K"]
            pbar.update(config["K"])

        # Bootstrap value
        with torch.no_grad():
            _, next_values = model(states)
        
        # Learn (Figure 2)
        al, cl, ent = agent.learn(next_values)

        # Gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        # =========================================================
        # Section 2.3: LOG EACH 1k STEPS
        # =========================================================
        if total_steps - last_log >= 1000:
            # Moyenner les épisodes terminés dans cette fenêtre
            if len(recent_episode_rewards) > 0:
                metrics["train_reward_mean"].append(np.mean(recent_episode_rewards))
                metrics["train_reward_std"].append(np.std(recent_episode_rewards))
                recent_episode_rewards.clear()
            else:
                # Si aucun épisode terminé, répéter dernière valeur
                if len(metrics["train_reward_mean"]) > 0:
                    metrics["train_reward_mean"].append(metrics["train_reward_mean"][-1])
                    metrics["train_reward_std"].append(metrics["train_reward_std"][-1])
                else:
                    metrics["train_reward_mean"].append(0.0)
                    metrics["train_reward_std"].append(0.0)
            
            metrics["train_steps"].append(total_steps)
            metrics["value_means"].append(np.mean(recent_values))
            metrics["actor_losses"].append(al)
            metrics["critic_losses"].append(cl)
            metrics["entropies"].append(ent)
            metrics["grad_norms"].append(total_norm)
            
            recent_values.clear()
            last_log = total_steps

        # =========================================================
        # Section 2.3: EVALUATE EVERY 20k STEPS
        # =========================================================
        if total_steps - last_eval >= config["eval_freq"]:
            # Section 2.3: "Plot value function on one full trajectory"
            # Stocker trajectoire aux étapes clés pour analyse
            store_traj = (total_steps % 100000 == 0) or \
                        (total_steps >= config["max_steps"] - config["eval_freq"])
            
            er, el, traj = evaluate(
                agent, config["env_name"], 
                episodes=10, 
                store_trajectory=store_traj
            )
            
            metrics["eval_steps"].append(total_steps)
            metrics["eval_rewards"].append(er)
            metrics["eval_lengths"].append(el)
            
            if traj is not None:
                metrics["trajectories"].append({
                    'step': total_steps,
                    'data': traj
                })
            
            last_eval = total_steps
            
            pbar.set_postfix({
                'eval_r': f'{er:.1f}',
                'train_r': f'{metrics["train_reward_mean"][-1]:.1f}' if metrics["train_reward_mean"] else '0',
                'actor_l': f'{al:.4f}',
                'critic_l': f'{cl:.4f}',
                'ent': f'{ent:.3f}'
            })

    pbar.close()
    envs.close()

    # =========================================================
    # SAUVEGARDE
    # =========================================================
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    out = PROJECT_ROOT / "results" / name
    out.mkdir(parents=True, exist_ok=True)

    save_metrics = {
        'seed': seed,
        'config': config,
        
        # Training (arrays)
        'train_steps': np.array(metrics['train_steps']),
        'train_reward_mean': np.array(metrics['train_reward_mean']),
        'train_reward_std': np.array(metrics['train_reward_std']),
        'actor_losses': np.array(metrics['actor_losses']),
        'critic_losses': np.array(metrics['critic_losses']),
        'entropies': np.array(metrics['entropies']),
        'value_means': np.array(metrics['value_means']),
        'grad_norms': np.array(metrics['grad_norms']),
        
        # Eval (arrays)
        'eval_steps': np.array(metrics['eval_steps']),
        'eval_rewards': np.array(metrics['eval_rewards']),
        'eval_lengths': np.array(metrics['eval_lengths']),
        
        # Trajectories (object array)
        'trajectories': np.array(metrics['trajectories'], dtype=object)
    }
    
    np.savez_compressed(out / f"seed{seed}.npz", **save_metrics)
    print(f"✅ Saved: {name} seed {seed} ({len(metrics['trajectories'])} trajectories)")


