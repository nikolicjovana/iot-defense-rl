"""
Deep Q-Network (DQN) Agent for IoT Intrusion Detection and Response
Implements DQN with experience replay and target network
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random


class DQNNetwork(nn.Module):
    """Deep Q-Network architecture"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128, 64]):
        """
        Initialize DQN network
        
        Args:
            state_dim: Dimension of state space (number of features)
            action_dim: Dimension of action space (number of actions)
            hidden_dims: List of hidden layer dimensions
        """
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state):
        """Forward pass"""
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN Agent for intrusion detection"""
    
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        memory_size=100000,
        batch_size=64,
        target_update_freq=100,
        device=None,
        hidden_dims=[128, 128, 64]
    ):
        """
        Initialize DQN Agent
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            lr: Learning rate
            gamma: Discount factor
            epsilon_start: Starting epsilon for epsilon-greedy
            epsilon_end: Ending epsilon for epsilon-greedy
            epsilon_decay: Epsilon decay rate
            memory_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency to update target network
            device: Device to run on (cuda/cpu)
            hidden_dims: Hidden layer dimensions
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Networks
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(memory_size)
    
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode (affects epsilon-greedy)
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, filepath):
        """Save agent model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load agent model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        print(f"Model loaded from {filepath}")
    
    def eval_mode(self):
        """Set agent to evaluation mode"""
        self.q_network.eval()
        self.target_network.eval()
    
    def train_mode(self):
        """Set agent to training mode"""
        self.q_network.train()
        self.target_network.train()


if __name__ == "__main__":
    # Test DQN Agent
    print("Testing DQN Agent...")
    
    state_dim = 46
    action_dim = 4
    
    agent = DQNAgent(state_dim, action_dim)
    
    # Test action selection
    state = np.random.randn(state_dim).astype(np.float32)
    action = agent.select_action(state)
    print(f"Selected action: {action}")
    
    # Test training
    for i in range(100):
        state = np.random.randn(state_dim).astype(np.float32)
        action = agent.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim).astype(np.float32)
        done = False
        
        agent.remember(state, action, reward, next_state, done)
        if len(agent.memory) >= agent.batch_size:
            loss = agent.train_step()
            if loss is not None:
                print(f"Step {i+1}, Loss: {loss:.4f}, Epsilon: {agent.epsilon:.4f}")
    
    print("\nDQN Agent test completed!")

