#plotting.py
import matplotlib.pyplot as plt

def plot_training_curves(rewards, actor_losses, critic_losses, label="Agent"):
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,3,1)
    plt.plot(rewards)
    plt.title(f"{label} - Rewards per episode")
    
    plt.subplot(1,3,2)
    plt.plot(actor_losses)
    plt.title(f"{label} - Actor Loss")
    
    plt.subplot(1,3,3)
    plt.plot(critic_losses)
    plt.title(f"{label} - Critic Loss")
    
    plt.tight_layout()
    plt.show()
