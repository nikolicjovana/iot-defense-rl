"""
Visualization script for training performance and agent behavior analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scipy import stats


def load_training_history(filepath='training_history.json'):
    """Load training history from JSON file"""
    with open(filepath, 'r') as f:
        history = json.load(f)
    return history


def plot_training_rewards(history, save_path='results/training_rewards.png', window=50):
    """Plot training rewards over episodes"""
    rewards = history['episode_rewards']
    episodes = np.arange(1, len(rewards) + 1)
    
    # Calculate moving average
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        moving_avg_episodes = np.arange(window, len(rewards) + 1)
    else:
        moving_avg = rewards
        moving_avg_episodes = episodes
    
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
    if len(moving_avg) > 0:
        plt.plot(moving_avg_episodes, moving_avg, color='red', linewidth=2, 
                label=f'Moving Average ({window} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards Over Episodes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Training rewards plot saved to {save_path}")
    plt.close()


def plot_training_loss(history, save_path='results/training_loss.png', window=50):
    """Plot training loss over episodes"""
    losses = [l for l in history['episode_losses'] if l is not None]
    if not losses:
        print("No loss data available for plotting")
        return
    
    episodes = np.arange(1, len(losses) + 1)
    
    # Calculate moving average
    if len(losses) >= window:
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        moving_avg_episodes = np.arange(window, len(losses) + 1)
    else:
        moving_avg = losses
        moving_avg_episodes = episodes
    
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, losses, alpha=0.3, color='blue', label='Episode Loss')
    if len(moving_avg) > 0:
        plt.plot(moving_avg_episodes, moving_avg, color='red', linewidth=2,
                label=f'Moving Average ({window} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Episodes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Use log scale for loss
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Training loss plot saved to {save_path}")
    plt.close()


def plot_detection_metrics(history, save_path='results/detection_metrics.png', window=50):
    """Plot detection rate and false positive rate over episodes"""
    detection_rates = history['episode_stats']['detection_rates']
    false_positive_rates = history['episode_stats']['false_positive_rates']
    episodes = np.arange(1, len(detection_rates) + 1)
    
    # Calculate moving averages
    if len(detection_rates) >= window:
        dr_avg = np.convolve(detection_rates, np.ones(window)/window, mode='valid')
        fpr_avg = np.convolve(false_positive_rates, np.ones(window)/window, mode='valid')
        avg_episodes = np.arange(window, len(detection_rates) + 1)
    else:
        dr_avg = detection_rates
        fpr_avg = false_positive_rates
        avg_episodes = episodes
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Detection rate
    ax1.plot(episodes, detection_rates, alpha=0.3, color='green', label='Episode Detection Rate')
    if len(dr_avg) > 0:
        ax1.plot(avg_episodes, dr_avg, color='darkgreen', linewidth=2,
                label=f'Moving Average ({window} episodes)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Detection Rate')
    ax1.set_title('Attack Detection Rate Over Episodes')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # False positive rate
    ax2.plot(episodes, false_positive_rates, alpha=0.3, color='red', label='Episode False Positive Rate')
    if len(fpr_avg) > 0:
        ax2.plot(avg_episodes, fpr_avg, color='darkred', linewidth=2,
                label=f'Moving Average ({window} episodes)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('False Positive Rate')
    ax2.set_title('False Positive Rate Over Episodes')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, max(0.1, max(false_positive_rates) * 1.1)])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Detection metrics plot saved to {save_path}")
    plt.close()


def plot_accuracy_over_time(history, save_path='results/accuracy_over_time.png', window=50):
    """Plot accuracy over episodes"""
    accuracies = history['episode_stats']['accuracies']
    episodes = np.arange(1, len(accuracies) + 1)
    
    # Calculate moving average
    if len(accuracies) >= window:
        moving_avg = np.convolve(accuracies, np.ones(window)/window, mode='valid')
        moving_avg_episodes = np.arange(window, len(accuracies) + 1)
    else:
        moving_avg = accuracies
        moving_avg_episodes = episodes
    
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, accuracies, alpha=0.3, color='purple', label='Episode Accuracy')
    if len(moving_avg) > 0:
        plt.plot(moving_avg_episodes, moving_avg, color='darkviolet', linewidth=2,
                label=f'Moving Average ({window} episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Episodes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Accuracy plot saved to {save_path}")
    plt.close()


def plot_epsilon_decay(history, save_path='results/epsilon_decay.png'):
    """Plot epsilon decay over episodes (if available)"""
    # Epsilon decay is tracked in the agent, not in history
    # This is a placeholder - you would need to log epsilon in training
    print("Epsilon decay plot not available (requires logging during training)")


def plot_learning_curves_summary(history, save_path='results/learning_curves_summary.png'):
    """Create a comprehensive summary of learning curves"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    episodes = np.arange(1, len(history['episode_rewards']) + 1)
    window = min(50, len(episodes) // 10)
    
    # Rewards
    rewards = history['episode_rewards']
    if len(rewards) >= window:
        rewards_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        rewards_episodes = np.arange(window, len(rewards) + 1)
        axes[0, 0].plot(rewards_episodes, rewards_avg, color='blue', linewidth=2)
    axes[0, 0].plot(episodes, rewards, alpha=0.3, color='blue')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    accuracies = history['episode_stats']['accuracies']
    if len(accuracies) >= window:
        acc_avg = np.convolve(accuracies, np.ones(window)/window, mode='valid')
        acc_episodes = np.arange(window, len(accuracies) + 1)
        axes[0, 1].plot(acc_episodes, acc_avg, color='purple', linewidth=2)
    axes[0, 1].plot(episodes, accuracies, alpha=0.3, color='purple')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3)
    
    # Detection Rate
    detection_rates = history['episode_stats']['detection_rates']
    if len(detection_rates) >= window:
        dr_avg = np.convolve(detection_rates, np.ones(window)/window, mode='valid')
        dr_episodes = np.arange(window, len(detection_rates) + 1)
        axes[1, 0].plot(dr_episodes, dr_avg, color='green', linewidth=2)
    axes[1, 0].plot(episodes, detection_rates, alpha=0.3, color='green')
    axes[1, 0].set_title('Detection Rate')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Detection Rate')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, alpha=0.3)
    
    # False Positive Rate
    false_positive_rates = history['episode_stats']['false_positive_rates']
    if len(false_positive_rates) >= window:
        fpr_avg = np.convolve(false_positive_rates, np.ones(window)/window, mode='valid')
        fpr_episodes = np.arange(window, len(false_positive_rates) + 1)
        axes[1, 1].plot(fpr_episodes, fpr_avg, color='red', linewidth=2)
    axes[1, 1].plot(episodes, false_positive_rates, alpha=0.3, color='red')
    axes[1, 1].set_title('False Positive Rate')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('False Positive Rate')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Learning curves summary saved to {save_path}")
    plt.close()


def analyze_agent_adaptation(history, save_path='results/agent_adaptation_analysis.png'):
    """Analyze how the agent adapts to new attack patterns"""
    # Divide training into phases
    num_phases = 4
    phase_size = len(history['episode_rewards']) // num_phases
    
    phases = []
    for i in range(num_phases):
        start_idx = i * phase_size
        end_idx = (i + 1) * phase_size if i < num_phases - 1 else len(history['episode_rewards'])
        
        phase_rewards = history['episode_rewards'][start_idx:end_idx]
        phase_accuracies = history['episode_stats']['accuracies'][start_idx:end_idx]
        phase_detection_rates = history['episode_stats']['detection_rates'][start_idx:end_idx]
        phase_false_positive_rates = history['episode_stats']['false_positive_rates'][start_idx:end_idx]
        
        phases.append({
            'phase': i + 1,
            'avg_reward': np.mean(phase_rewards),
            'std_reward': np.std(phase_rewards),
            'avg_accuracy': np.mean(phase_accuracies),
            'avg_detection_rate': np.mean(phase_detection_rates),
            'avg_false_positive_rate': np.mean(phase_false_positive_rates)
        })
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    phase_nums = [p['phase'] for p in phases]
    
    # Rewards by phase
    avg_rewards = [p['avg_reward'] for p in phases]
    std_rewards = [p['std_reward'] for p in phases]
    axes[0, 0].bar(phase_nums, avg_rewards, yerr=std_rewards, capsize=5, color='blue', alpha=0.7)
    axes[0, 0].set_title('Average Reward by Training Phase')
    axes[0, 0].set_xlabel('Training Phase')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Accuracy by phase
    avg_accuracies = [p['avg_accuracy'] for p in phases]
    axes[0, 1].bar(phase_nums, avg_accuracies, color='purple', alpha=0.7)
    axes[0, 1].set_title('Average Accuracy by Training Phase')
    axes[0, 1].set_xlabel('Training Phase')
    axes[0, 1].set_ylabel('Average Accuracy')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Detection rate by phase
    avg_detection_rates = [p['avg_detection_rate'] for p in phases]
    axes[1, 0].bar(phase_nums, avg_detection_rates, color='green', alpha=0.7)
    axes[1, 0].set_title('Average Detection Rate by Training Phase')
    axes[1, 0].set_xlabel('Training Phase')
    axes[1, 0].set_ylabel('Average Detection Rate')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # False positive rate by phase
    avg_false_positive_rates = [p['avg_false_positive_rate'] for p in phases]
    axes[1, 1].bar(phase_nums, avg_false_positive_rates, color='red', alpha=0.7)
    axes[1, 1].set_title('Average False Positive Rate by Training Phase')
    axes[1, 1].set_xlabel('Training Phase')
    axes[1, 1].set_ylabel('Average False Positive Rate')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Agent adaptation analysis saved to {save_path}")
    plt.close()
    
    # Print adaptation analysis
    print("\n" + "=" * 60)
    print("AGENT ADAPTATION ANALYSIS")
    print("=" * 60)
    for phase in phases:
        print(f"\nPhase {phase['phase']}:")
        print(f"  Average Reward: {phase['avg_reward']:.2f} ± {phase['std_reward']:.2f}")
        print(f"  Average Accuracy: {phase['avg_accuracy']:.4f} ({phase['avg_accuracy']*100:.2f}%)")
        print(f"  Average Detection Rate: {phase['avg_detection_rate']:.4f} ({phase['avg_detection_rate']*100:.2f}%)")
        print(f"  Average False Positive Rate: {phase['avg_false_positive_rate']:.4f} ({phase['avg_false_positive_rate']*100:.2f}%)")
    
    # Calculate improvement
    if len(phases) > 1:
        reward_improvement = phases[-1]['avg_reward'] - phases[0]['avg_reward']
        accuracy_improvement = phases[-1]['avg_accuracy'] - phases[0]['avg_accuracy']
        detection_improvement = phases[-1]['avg_detection_rate'] - phases[0]['avg_detection_rate']
        
        print("\n" + "=" * 60)
        print("IMPROVEMENT FROM PHASE 1 TO PHASE 4")
        print("=" * 60)
        print(f"  Reward: {reward_improvement:+.2f}")
        print(f"  Accuracy: {accuracy_improvement:+.4f} ({accuracy_improvement*100:+.2f}%)")
        print(f"  Detection Rate: {detection_improvement:+.4f} ({detection_improvement*100:+.2f}%)")
    print("=" * 60)


def main():
    """Main visualization function"""
    print("=" * 60)
    print("RL-based IoT Intrusion Detection System - Visualization")
    print("=" * 60)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load training history
    history_file = 'training_history.json'
    if not os.path.exists(history_file):
        print(f"Error: Training history file not found at {history_file}")
        print("Please train the model first using train.py")
        return
    
    print(f"\nLoading training history from {history_file}...")
    history = load_training_history(history_file)
    
    print(f"Loaded {len(history['episode_rewards'])} episodes")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_training_rewards(history)
    plot_training_loss(history)
    plot_detection_metrics(history)
    plot_accuracy_over_time(history)
    plot_learning_curves_summary(history)
    analyze_agent_adaptation(history)
    
    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)
    print("\nVisualizations saved in the 'results' directory:")
    print("  - training_rewards.png")
    print("  - training_loss.png")
    print("  - detection_metrics.png")
    print("  - accuracy_over_time.png")
    print("  - learning_curves_summary.png")
    print("  - agent_adaptation_analysis.png")


if __name__ == "__main__":
    main()

