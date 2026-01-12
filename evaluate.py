import torch
import numpy as np

def evaluate(agent, env, num_episodes=10):
    """
    Evaluate agent greedily for num_episodes
    
    Args:
        agent: A2CAgent instance
        env: Gymnasium environment
        num_episodes: Number of episodes to evaluate
    
    Returns:
        Dictionary with evaluation metrics
    """
    returns = []
    episode_lengths = []
    value_trajectories = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_return = 0
        episode_length = 0
        terminated = False
        truncated = False
        
        episode_values = []
        
        while not (terminated or truncated):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Select greedy action
            action = agent.select_action_greedy(state_tensor)
            
            # Get value estimate for logging
            with torch.no_grad():
                value = agent.critic(state_tensor).item()
                episode_values.append(value)
            
            # Step in environment
            state, reward, terminated, truncated, _ = env.step(action)
            
            episode_return += reward
            episode_length += 1
        
        returns.append(episode_return)
        episode_lengths.append(episode_length)
        value_trajectories.append(episode_values)
    
    # Compute statistics
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    mean_length = np.mean(episode_lengths)
    
    # Get one representative value trajectory (from first episode)
    value_trajectory = value_trajectories[0] if value_trajectories else []
    
    return {
        'mean_return': mean_return,
        'std_return': std_return,
        'mean_length': mean_length,
        'returns': returns,
        'episode_lengths': episode_lengths,
        'value_trajectory': value_trajectory,
        'all_value_trajectories': value_trajectories
    }