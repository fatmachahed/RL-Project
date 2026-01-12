import numpy as np
import torch
import gymnasium as gym

class Rollout:
    """Handles K parallel environments and n-step data collection"""
    
    def __init__(self, env_fn, num_workers=1, n_steps=1):
        """
        Args:
            env_fn: Function that creates a single environment
            num_workers: K parallel environments
            n_steps: n-step returns
        """
        self.num_workers = num_workers
        self.n_steps = n_steps
        self.episode_returns = []  # Track completed episode returns
        
        # Create environments
        if num_workers == 1:
            self.env = env_fn()
            self.is_vectorized = False
            state, _ = self.env.reset()
            self.states = np.array([state])  # Shape: (1, state_dim)
        else:
            # Use vectorized environments for K > 1
            self.env = gym.vector.AsyncVectorEnv([
                env_fn for _ in range(num_workers)
            ])
            self.is_vectorized = True
            self.states, _ = self.env.reset()  # Shape: (K, state_dim)
        
        # Track episode statistics
        self.episode_rewards = np.zeros(num_workers)
        self.episode_lengths = np.zeros(num_workers)
    
    def collect_rollout(self, agent, gamma=0.99):
        """
        Collect n steps from K workers
        
        Returns:
            batch: Dictionary with:
                - states: (n, K, state_dim)
                - actions: (n, K)
                - rewards: (n, K)
                - next_states: (n, K, state_dim)
                - dones: (n, K) - True if episode ended
                - values: (n, K) - Value estimates
                - returns: (n, K) - n-step returns for each state
        """
        states_list = []
        actions_list = []
        rewards_list = []
        next_states_list = []
        dones_list = []
        values_list = []
        terminated_list = []
        truncated_list = []
        
        for step in range(self.n_steps):
            # Agent selects actions for all K environments
            with torch.no_grad():
                states_tensor = torch.FloatTensor(self.states)
                actions, log_probs, values = agent.select_action(states_tensor)
            
            # Convert to numpy for environment
            if self.is_vectorized:
                actions_np = actions.cpu().numpy()
            else:
                actions_np = actions.cpu().numpy()[0]
            
            # Step in environment(s)
            next_states, rewards, terminated, truncated, infos = self.env.step(actions_np)
            
            # Handle single env case
            if not self.is_vectorized:
                next_states = np.array([next_states])
                rewards = np.array([rewards])
                terminated = np.array([terminated])
                truncated = np.array([truncated])
            
            dones = terminated | truncated
            
            # Track episode statistics
            self.episode_rewards += rewards
            self.episode_lengths += 1
            
            # Log completed episodes
            for i in range(self.num_workers):
                if dones[i]:
                    self.episode_returns.append(self.episode_rewards[i])
                    self.episode_rewards[i] = 0
                    self.episode_lengths[i] = 0
            
            # Store transition
            states_list.append(self.states.copy())
            actions_list.append(actions.cpu().numpy())
            rewards_list.append(rewards)
            next_states_list.append(next_states.copy())
            dones_list.append(dones)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            values_list.append(values.cpu().numpy())
            
            # Update current state
            self.states = next_states
        
        # Convert lists to arrays
        # Shape: (n_steps, num_workers, ...)
        states = np.array(states_list)
        actions = np.array(actions_list)
        rewards = np.array(rewards_list)
        next_states = np.array(next_states_list)
        dones = np.array(dones_list)
        terminated = np.array(terminated_list)
        truncated = np.array(truncated_list)
        values = np.array(values_list)
        
        # Compute n-step returns with proper bootstrapping
        returns = self._compute_nstep_returns(
            rewards, values, dones, terminated, truncated, 
            next_states[-1], agent, gamma
        )
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'values': values,
            'returns': returns,
            'terminated': terminated,
            'truncated': truncated
        }
    
    def _compute_nstep_returns(self, rewards, values, dones, terminated, 
                               truncated, final_states, agent, gamma):
        """
        Compute n-step returns with proper bootstrapping
        
        CRITICAL: Bootstrap at truncation, don't bootstrap at termination
        """
        n_steps, num_workers = rewards.shape
        returns = np.zeros_like(rewards)
        
        # Get bootstrap value for final state
        with torch.no_grad():
            final_states_tensor = torch.FloatTensor(final_states)
            _, _, next_values = agent.select_action(final_states_tensor)
            next_values = next_values.cpu().numpy()
        
        # Compute returns backwards
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                # Last step: bootstrap from final state if truncated
                # Don't bootstrap if terminated (reached terminal state)
                bootstrap_value = next_values * truncated[-1]
            else:
                # Middle steps: bootstrap from next value if not done
                bootstrap_value = returns[t + 1] * (~dones[t])
            
            returns[t] = rewards[t] + gamma * bootstrap_value
        
        return returns
    
    def close(self):
        """Close all environments"""
        self.env.close()