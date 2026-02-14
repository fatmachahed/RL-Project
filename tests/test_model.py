import torch
import torch.nn as nn
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ActorCritic

def test_model():
    """Test unitaire pour vérifier l'architecture du modèle"""
    print("Testing ActorCritic model...")

    # Paramètres CartPole-v1
    state_dim = 4
    action_dim = 2
    hidden_size = 64

    # Créer modèle
    model = ActorCritic(state_dim, action_dim, hidden_size)

    # Test forward pass - batch
    batch_size = 5
    dummy_state = torch.randn(batch_size, state_dim)
    logits, value = model(dummy_state)

    assert logits.shape == (batch_size, action_dim), f"Expected logits shape ({batch_size}, {action_dim}), got {logits.shape}"
    assert value.shape == (batch_size,), f"Expected value shape ({batch_size},), got {value.shape}"

    # Test forward pass - single state
    single_state = torch.randn(state_dim)
    if single_state.dim() == 1:
        single_state = single_state.unsqueeze(0)  # batch size = 1
    logits_single, value_single = model(single_state)
    assert logits_single.shape == (1, action_dim), f"Expected single logits shape (1, {action_dim}), got {logits_single.shape}"
    assert value_single.shape == (1,), f"Expected single value shape (1,), got {value_single.shape}"

    # Exemple simple d'action via softmax
    probs = torch.softmax(logits_single, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    entropy = dist.entropy()

    print("✓ Model test passed!")
    print(f"  Input shape: {dummy_state.shape}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Sampled action: {action.item()}")
    print(f"  Log probability: {log_prob.item():.4f}")
    print(f"  Entropy: {entropy.item():.4f}")
    print(f"  State value: {value_single.item():.4f}")

    return model

if __name__ == "__main__":
    test_model()
