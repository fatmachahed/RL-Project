# Reinforcement Learning Project - A2C on CartPole-v1

## 📌 Overview

This project implements **Advantage Actor-Critic (A2C)** agents with various configurations to explore the impact of:
- **Parallel workers (K)**: Multiple environments running simultaneously
- **n-step returns**: Multi-step bootstrapping for gradient estimation
- **Stochastic rewards**: Reward masking to simulate real-world uncertainty

The agents are trained on the **CartPole-v1** environment from Gymnasium, with comprehensive performance analysis and visualizations.

---

## 🎯 Project Goals

1. Implement a correct A2C algorithm following academic specifications
2. Compare 5 different agent configurations
3. Analyze the effects of parallel workers and n-step returns
4. Study learning stability under stochastic reward conditions
5. Generate publication-ready visualizations

---

## 🗂 Project Structure
```
RL-Project/
│
├── src/
│   ├── model.py           # ActorCritic neural network
│   ├── a2c_agent.py       # A2C agent with n-step returns
│   ├── evaluate.py        # Evaluation and trajectory collection
│   └── train.py           # Main training loop
│
├── configs/
│   ├── agent0.yaml        # K=1, n=1, deterministic
│   ├── agent1.yaml        # K=1, n=1, stochastic
│   ├── agent2.yaml        # K=6, n=1, stochastic
│   ├── agent3.yaml        # K=1, n=6, stochastic
│   └── agent4.yaml        # K=6, n=6, stochastic
│
├── tests/
│   ├── run_all_agents.py  # Train all agents
│   └── test_graphs.py     # Generate visualizations
│
├── results/               # Training metrics (NPZ files)
├── plots_0001/           # plots with  lr_critic: 0.001 
├── plots_0003/           # plots with lr_critic: 0.003
├── notebooks/            # Analysis notebooks
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Create a virtual environment
```bash
python -m venv venv
```

**Activate it:**
- Windows: `venv\Scripts\activate`
- Linux/Mac: `source venv/bin/activate`

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

**Required libraries:**
- `torch` >= 2.0.0
- `gymnasium` >= 0.29.0
- `numpy` >= 1.24.0
- `matplotlib` >= 3.7.0
- `seaborn` >= 0.12.0
- `pyyaml` >= 6.0
- `tqdm` >= 4.65.0

---

## 🚀 Usage

### Train all agents (3 seeds each)
```bash
python -m tests.run_all_agents
```

This will train 5 agents × 3 seeds = 15 runs and save metrics in `results/`.

### Generate visualizations
```bash
python -m tests.test_graphs
```

All plots will be saved in `plots_0001/` and `plots_0003/`  with timestamps.

### Train a single agent
```bash
python -m src.train --config configs/agent0.yaml
```

---

## 🤖 Agent Configurations

| Agent   | K (workers) | n (steps) | Stochastic | lr_actor | lr_critic | Description |
|---------|-------------|-----------|------------|----------|-----------|-------------|
| **Agent0** | 1 | 1 | ❌ | 1e-5 | 1e-3 | Baseline (deterministic) |
| **Agent1** | 1 | 1 | ✅ | 1e-5 | 1e-3 | Stochastic rewards (90% masking) |
| **Agent2** | 6 | 1 | ✅ | 1e-5 | 1e-3 | Parallel workers |
| **Agent3** | 1 | 6 | ✅ | 1e-5 | 1e-3 | Multi-step returns |
| **Agent4** | 6 | 6 | ✅ | 3e-4 | 3e-3 | Combined (best performance) |

### Common hyperparameters:
- **Environment**: CartPole-v1
- **Max steps**: 500,000
- **Discount factor (γ)**: 0.99
- **Entropy coefficient**: 0.01
- **Evaluation frequency**: 20,000 steps
- **Hidden layers**: 2 × 64 neurons (Tanh activation)
- **Optimizer**: Adam (separate LR for actor/critic)

---

## 🧠 Algorithm Details

### Network Architecture
```
Input (4D state) 
    ↓
Linear(4 → 64) + Tanh
    ↓
Linear(64 → 64) + Tanh
    ↓
    ├─→ Actor: Linear(64 → 2)   [action logits]
    └─→ Critic: Linear(64 → 1)  [state value]
```

### A2C Update Rule
```
Advantage: A(s,a) = R - V(s)
Actor Loss: -log π(a|s) × A(s,a)
Critic Loss: 0.5 × (R - V(s))²
Entropy: -Σ π(a|s) log π(a|s)

Total Loss = Actor Loss + Critic Loss - entropy_coef × Entropy
```

### Key Implementation Features

✅ **Correct bootstrapping**: Distinguishes truncation vs. termination  
✅ **n-step returns**: Bootstraps after n steps for variance reduction  
✅ **Parallel workers**: K synchronized environments for stable gradients  
✅ **Stochastic rewards**: 90% masking (agents 1-4) to simulate uncertainty  
✅ **Gradient clipping**: Max norm = 0.5 for training stability  

---

## 📊 Generated Visualizations

The project produces **11 comprehensive plots**:

1. **Learning Curves** - Evaluation rewards (mean ± std over 3 seeds)
2. **Training Rewards** - Episode returns during training
3. **Actor Loss** - Policy gradient loss evolution
4. **Critic Loss** - Value function MSE evolution
5. **Value Function Mean** - Average predicted values
6. **Trajectory Values** - V(s) along full episodes
7. **Entropy** - Policy entropy over training
8. **Performance Comparison** - Boxplots of final rewards
9. **Training Stability** - Coefficient of variation analysis
10. **Correlation Heatmap** - Metric relationships
11. **Convergence Speed** - Steps to reach target performance

All plots include:
- Mean curves with ±1 std shaded areas (across 3 seeds)
- Smooth curves (moving average for clarity)
- Publication-ready quality (high DPI, clear labels)

---

## 📈 Expected Results

### Agent Performance Ranking (best → worst):
```
Agent4 (K=6, n=6) > Agent2 (K=6) ≈ Agent3 (n=6) > Agent0 > Agent1
```

### Key Findings:

- **Agent0**: Should reach 500 reward (optimal policy)
- **Agent1**: Slower/unstable due to stochastic rewards
- **Agent2**: Faster convergence (parallel workers reduce variance)
- **Agent3**: More stable (n-step returns smooth gradients)
- **Agent4**: Best of both worlds (fastest + most stable)

### Value Function Convergence:

With correct bootstrapping:
- **Deterministic**: V(s) ≈ 495 at optimal policy
- **Stochastic (90% mask)**: V(s) ≈ 50 at optimal policy

---

## 🔬 Experimental Protocol

### Training:
- 500,000 environment steps per agent
- Log metrics every 1,000 steps
- Evaluate every 20,000 steps
- 3 random seeds per configuration

### Evaluation:
- 10 episodes per evaluation
- Greedy policy (argmax action)
- Fresh environment (no training contamination)
- Full trajectory storage at 100k, 200k, 300k, 400k, 500k steps

---

## 🛠 Troubleshooting

### Issue: Agent0 doesn't learn (reward ~10)

**Solution**: Learning rate too low. Try `lr_actor: 3e-4` instead of `1e-5`.

### Issue: High variance in training

**Solution**: Increase K (workers) or n (steps) to reduce gradient variance.



## 👥 Team

**Author**: Fatma Chahed  and Dhif Aziz 

**Program**: Business Intelligence & Data Science  
**Institution**: Université Paris Dauphine (Tunis)  
**Course**: Reinforcement Learning (S.Moalla)  
**Date**: February 2026

---
**⭐ If you find this project useful, please consider starring it!**
