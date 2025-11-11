"""
Training script for RL-based IoT Intrusion Detection System
"""

import numpy as np
import torch
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from preprocessing import DataPreprocessor
from rl_environment import IoTIntrusionDetectionEnv
from dqn_agent import DQNAgent


class TrainingLogger:
    """Logger for training metrics"""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_losses = []
        self.episode_stats = {
            'detection_rates': [],
            'false_positive_rates': [],
            'accuracies': [],
            'step_counts': []
        }
        self.training_history = []
    
    def log_episode(self, reward, loss, stats, episode):
        """Log episode results"""
        self.episode_rewards.append(reward)
        if loss is not None:
            self.episode_losses.append(loss)
        
        self.episode_stats['detection_rates'].append(stats.get('detection_rate', 0))
        self.episode_stats['false_positive_rates'].append(stats.get('false_positive_rate', 0))
        self.episode_stats['step_counts'].append(stats.get('total_attacks', 0) + stats.get('total_benign', 0))
        
        # Calculate accuracy
        total = stats.get('correct_actions', 0) + stats.get('false_positives', 0) + stats.get('false_negatives', 0)
        if total > 0:
            accuracy = stats.get('correct_actions', 0) / total
        else:
            accuracy = 0
        self.episode_stats['accuracies'].append(accuracy)
        
        self.training_history.append({
            'episode': episode,
            'reward': reward,
            'loss': loss,
            'stats': stats.copy()
        })
    
    def get_recent_stats(self, window=100):
        """Get recent statistics over a window"""
        if len(self.episode_rewards) < window:
            window = len(self.episode_rewards)
        
        recent_rewards = self.episode_rewards[-window:]
        recent_accuracies = self.episode_stats['accuracies'][-window:]
        recent_detection_rates = self.episode_stats['detection_rates'][-window:]
        
        return {
            'avg_reward': np.mean(recent_rewards),
            'avg_accuracy': np.mean(recent_accuracies),
            'avg_detection_rate': np.mean(recent_detection_rates),
            'std_reward': np.std(recent_rewards),
            'std_accuracy': np.std(recent_accuracies)
        }
    
    def save(self, filepath):
        """Save training history"""
        with open(filepath, 'w') as f:
            json.dump({
                'episode_rewards': self.episode_rewards,
                'episode_losses': self.episode_losses,
                'episode_stats': {k: v for k, v in self.episode_stats.items()},
                'training_history': self.training_history
            }, f, indent=2)
        print(f"Training history saved to {filepath}")


def train_agent(
    agent,
    env,
    num_episodes=1000,
    max_steps_per_episode=1000,
    save_freq=100,
    save_dir='models',
    logger=None
):
    """
    Train the RL agent
    
    Args:
        agent: DQN agent
        env: RL environment
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        save_freq: Frequency to save model
        save_dir: Directory to save models
        logger: Training logger
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nStarting training for {num_episodes} episodes...")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print(f"Agent epsilon: {agent.epsilon:.4f}")
    
    start_time = time.time()
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        obs, info = env.reset()
        episode_reward = 0
        episode_losses = []
        steps = 0
        
        for step in range(max_steps_per_episode):
            # Select action
            action = agent.select_action(obs, training=True)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            
            # Store experience
            done = terminated or truncated
            agent.remember(obs, action, reward, next_obs, done)
            
            # Train agent
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            
            episode_reward += reward
            obs = next_obs
            steps += 1
            
            if done:
                break
        
        # Log episode
        episode_stats = env.get_episode_stats()
        avg_loss = np.mean(episode_losses) if episode_losses else None
        
        if logger:
            logger.log_episode(episode_reward, avg_loss, episode_stats, episode)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            recent_stats = logger.get_recent_stats(window=10) if logger else {}
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Loss: {avg_loss:.4f}" if avg_loss else "  Loss: N/A")
            print(f"  Detection Rate: {episode_stats.get('detection_rate', 0):.2%}")
            print(f"  False Positive Rate: {episode_stats.get('false_positive_rate', 0):.2%}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            if recent_stats:
                print(f"  Recent Avg Reward: {recent_stats['avg_reward']:.2f}")
                print(f"  Recent Avg Accuracy: {recent_stats['avg_accuracy']:.2%}")
        
        # Save model
        if (episode + 1) % save_freq == 0:
            model_path = os.path.join(save_dir, f'dqn_agent_episode_{episode + 1}.pth')
            agent.save(model_path)
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes")
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'dqn_agent_final.pth')
    agent.save(final_model_path)
    
    return agent, logger


def main():
    """Main training function"""
    # Configuration
    config = {
        'sample_size': None,  # Set to None for full dataset, or a number for sampling
        'num_episodes': 500,
        'max_steps_per_episode': 500,
        'lr': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 64,
        'memory_size': 100000,
        'target_update_freq': 100,
        'save_freq': 50
    }
    
    print("=" * 60)
    print("RL-based IoT Intrusion Detection System - Training")
    print("=" * 60)
    
    # Step 1: Preprocess data
    print("\n[Step 1] Preprocessing data...")
    preprocessor = DataPreprocessor(
        train_path='CICIOT23/train/train.csv',
        val_path='CICIOT23/validation/validation.csv',
        test_path='CICIOT23/test/test.csv'
    )
    
    train_df, val_df, test_df = preprocessor.load_data(sample_size=config['sample_size'])
    data = preprocessor.preprocess(train_df, val_df, test_df)
    
    # Save training mean for missing value imputation during evaluation
    feature_cols = [col for col in train_df.columns if col != 'label']
    train_mean = train_df[feature_cols].mean()
    preprocessor.save_preprocessor('preprocessor.pkl', train_mean=train_mean)
    
    # Step 2: Create environment
    print("\n[Step 2] Creating RL environment...")
    # Get benign label index
    benign_label = preprocessor.label_encoder.transform(['BenignTraffic'])[0]
    print(f"Benign traffic label index: {benign_label}")
    
    env = IoTIntrusionDetectionEnv(
        X_data=data['X_train'],
        y_data=data['y_train'],
        max_steps=config['max_steps_per_episode'],
        benign_label_idx=benign_label
    )
    
    # Step 3: Create agent
    print("\n[Step 3] Creating DQN agent...")
    agent = DQNAgent(
        state_dim=data['num_features'],
        action_dim=4,
        lr=config['lr'],
        gamma=config['gamma'],
        epsilon_start=config['epsilon_start'],
        epsilon_end=config['epsilon_end'],
        epsilon_decay=config['epsilon_decay'],
        batch_size=config['batch_size'],
        memory_size=config['memory_size'],
        target_update_freq=config['target_update_freq']
    )
    
    # Step 4: Train agent
    print("\n[Step 4] Training agent...")
    logger = TrainingLogger()
    
    agent, logger = train_agent(
        agent=agent,
        env=env,
        num_episodes=config['num_episodes'],
        max_steps_per_episode=config['max_steps_per_episode'],
        save_freq=config['save_freq'],
        save_dir='models',
        logger=logger
    )
    
    # Step 5: Save training history
    print("\n[Step 5] Saving training history...")
    logger.save('training_history.json')
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"\nFinal statistics:")
    recent_stats = logger.get_recent_stats(window=50)
    for key, value in recent_stats.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()

