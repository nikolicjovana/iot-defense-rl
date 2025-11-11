"""
Reinforcement Learning Environment for IoT Intrusion Detection and Response
Simulates network traffic flows and provides rewards based on detection accuracy and response actions
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import random


class IoTIntrusionDetectionEnv(gym.Env):
    """
    RL Environment for adaptive intrusion detection and response in IoT networks
    
    State Space: Network traffic features (46 features from CICIoT2023)
    Action Space: Discrete actions [0: Allow, 1: Block, 2: Isolate, 3: Regulate]
    Reward: Based on correct detection, false positives, and response effectiveness
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, X_data, y_data, max_steps=1000, reward_config=None, benign_label_idx=1):
        """
        Initialize the environment
        
        Args:
            X_data: Feature matrix (numpy array)
            y_data: Labels (numpy array with encoded labels)
            max_steps: Maximum steps per episode
            reward_config: Dictionary with reward parameters
            benign_label_idx: Index of benign traffic label (default: 1 for BenignTraffic)
        """
        super(IoTIntrusionDetectionEnv, self).__init__()
        
        self.X_data = X_data
        self.y_data = y_data
        self.max_steps = max_steps
        self.num_samples = len(X_data)
        self.benign_label_idx = benign_label_idx
        
        # Determine if label is attack (1) or benign (0)
        # BenignTraffic is typically at index 1 in the CICIoT2023 dataset
        self.is_attack = (y_data != benign_label_idx).astype(int)
        
        # Reward configuration
        default_reward_config = {
            'correct_detection': 10.0,      # Reward for correctly detecting attack
            'correct_benign': 2.0,          # Reward for correctly allowing benign
            'false_positive': -5.0,         # Penalty for blocking benign traffic
            'false_negative': -20.0,        # Penalty for missing an attack
            'block_attack': 5.0,            # Bonus for blocking attack
            'isolate_attack': 8.0,          # Bonus for isolating attack
            'regulate_attack': 3.0,         # Bonus for regulating attack
            'unnecessary_action': -2.0,     # Penalty for unnecessary action on benign
            'missed_critical': -30.0        # Extra penalty for missing critical attacks
        }
        self.reward_config = reward_config or default_reward_config
        
        # Define action and observation spaces
        # Actions: 0=Allow, 1=Block, 2=Isolate, 3=Regulate
        self.action_space = spaces.Discrete(4)
        
        # State space: features from network traffic
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(X_data.shape[1],), 
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.current_index = 0
        self.episode_reward = 0
        self.episode_stats = {
            'total_attacks': 0,
            'detected_attacks': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'correct_actions': 0,
            'total_benign': 0
        }
        
        # Buffer for recent traffic (for context)
        self.traffic_buffer = deque(maxlen=10)
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Randomly select starting point in dataset
        self.current_index = self.np_random.integers(0, self.num_samples)
        self.current_step = 0
        self.episode_reward = 0
        self.episode_stats = {
            'total_attacks': 0,
            'detected_attacks': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'correct_actions': 0,
            'total_benign': 0
        }
        self.traffic_buffer.clear()
        
        # Get initial observation
        observation = self.X_data[self.current_index].astype(np.float32)
        info = {
            'is_attack': bool(self.is_attack[self.current_index]),
            'label': int(self.y_data[self.current_index])
        }
        
        return observation, info
    
    def step(self, action):
        """Execute one step in the environment"""
        if self.current_step >= self.max_steps:
            # Episode ended due to max steps
            terminated = True
            truncated = False
        elif self.current_index >= self.num_samples - 1:
            # Reached end of dataset
            terminated = True
            truncated = True
        else:
            terminated = False
            truncated = False
        
        # Get current traffic flow
        is_attack = self.is_attack[self.current_index]
        label = self.y_data[self.current_index]
        observation = self.X_data[self.current_index].astype(np.float32)
        
        # Update stats
        if is_attack:
            self.episode_stats['total_attacks'] += 1
        else:
            self.episode_stats['total_benign'] += 1
        
        # Calculate reward based on action and ground truth
        reward = self._calculate_reward(action, is_attack, label)
        self.episode_reward += reward
        
        # Update episode statistics
        self._update_stats(action, is_attack)
        
        # Move to next sample
        self.current_index = (self.current_index + 1) % self.num_samples
        self.current_step += 1
        
        # Prepare info
        info = {
            'is_attack': bool(is_attack),
            'label': int(label),
            'action_taken': int(action),
            'step': self.current_step,
            'episode_stats': self.episode_stats.copy()
        }
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, action, is_attack, label):
        """Calculate reward based on action and ground truth"""
        reward = 0.0
        
        if is_attack:
            # This is an attack
            if action == 0:  # Allow (WRONG - false negative)
                reward += self.reward_config['false_negative']
                # Extra penalty for critical attacks (you can customize this)
                if label in [0, 1, 2]:  # Example: first few attack types might be critical
                    reward += self.reward_config['missed_critical']
            elif action == 1:  # Block (CORRECT)
                reward += self.reward_config['correct_detection']
                reward += self.reward_config['block_attack']
            elif action == 2:  # Isolate (CORRECT - even better)
                reward += self.reward_config['correct_detection']
                reward += self.reward_config['isolate_attack']
            elif action == 3:  # Regulate (CORRECT - moderate response)
                reward += self.reward_config['correct_detection']
                reward += self.reward_config['regulate_attack']
        else:
            # This is benign traffic
            if action == 0:  # Allow (CORRECT)
                reward += self.reward_config['correct_benign']
            else:  # Any action on benign (WRONG - false positive)
                reward += self.reward_config['false_positive']
                if action in [1, 2]:  # Block or Isolate
                    reward += self.reward_config['unnecessary_action']
        
        return reward
    
    def _update_stats(self, action, is_attack):
        """Update episode statistics"""
        if is_attack:
            if action != 0:  # Detected attack
                self.episode_stats['detected_attacks'] += 1
                self.episode_stats['correct_actions'] += 1
            else:  # Missed attack
                self.episode_stats['false_negatives'] += 1
        else:
            if action == 0:  # Correctly allowed benign
                self.episode_stats['correct_actions'] += 1
            else:  # False positive
                self.episode_stats['false_positives'] += 1
    
    def get_episode_stats(self):
        """Get statistics for the current episode"""
        stats = self.episode_stats.copy()
        if stats['total_attacks'] > 0:
            stats['detection_rate'] = stats['detected_attacks'] / stats['total_attacks']
        else:
            stats['detection_rate'] = 0.0
        
        if stats['total_benign'] > 0:
            stats['false_positive_rate'] = stats['false_positives'] / stats['total_benign']
        else:
            stats['false_positive_rate'] = 0.0
        
        return stats
    
    def render(self):
        """Render the environment (optional)"""
        pass


class TrafficFlowGenerator:
    """Utility class to generate traffic flows in batches for training"""
    
    def __init__(self, X_data, y_data, batch_size=32, shuffle=True):
        self.X_data = X_data
        self.y_data = y_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X_data))
        if shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0
    
    def get_batch(self):
        """Get a batch of traffic flows"""
        if self.current_idx + self.batch_size > len(self.indices):
            if self.shuffle:
                np.random.shuffle(self.indices)
            self.current_idx = 0
        
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        
        return self.X_data[batch_indices], self.y_data[batch_indices]
    
    def reset(self):
        """Reset the generator"""
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)


if __name__ == "__main__":
    # Test environment
    print("Testing IoT Intrusion Detection Environment...")
    
    # Create dummy data
    X_dummy = np.random.randn(1000, 46).astype(np.float32)
    y_dummy = np.random.randint(0, 2, 1000)  # Binary: 0=benign, 1=attack
    
    env = IoTIntrusionDetectionEnv(X_dummy, y_dummy, max_steps=100)
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Test steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Attack={info['is_attack']}")
    
    print("\nEnvironment test completed!")

