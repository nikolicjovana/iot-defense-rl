"""
Evaluation script for RL-based IoT Intrusion Detection System
Computes accuracy, precision, recall, F1-score, and response latency
"""

import numpy as np
import torch
import time
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def _to_serializable(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

from preprocessing import DataPreprocessor
from rl_environment import IoTIntrusionDetectionEnv
from dqn_agent import DQNAgent


class ModelEvaluator:
    """Evaluator for RL agent performance"""
    
    def __init__(self, agent, env, label_encoder):
        self.agent = agent
        self.env = env
        self.label_encoder = label_encoder
        self.agent.eval_mode()
    
    def evaluate(self, num_episodes=100, max_steps=1000, verbose=True):
        """
        Evaluate agent performance
        
        Args:
            num_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode
            verbose: Print detailed results
        """
        all_predictions = []
        all_labels = []
        all_actions = []
        all_rewards = []
        all_latencies = []
        episode_stats = []
        
        print(f"\nEvaluating agent over {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_predictions = []
            episode_labels = []
            episode_actions = []
            episode_latencies = []
            
            for step in range(max_steps):
                # Measure latency
                start_time = time.time()
                action = self.agent.select_action(obs, training=False)
                latency = time.time() - start_time
                
                next_obs, reward, terminated, truncated, step_info = self.env.step(action)
                
                # Store results
                is_attack = step_info['is_attack']
                label = step_info['label']
                
                # Convert action to prediction: 0=benign, 1=attack
                prediction = 1 if action != 0 else 0
                
                episode_predictions.append(prediction)
                episode_labels.append(1 if is_attack else 0)
                episode_actions.append(action)
                episode_latencies.append(latency)
                episode_reward += reward
                
                obs = next_obs
                
                if terminated or truncated:
                    break
            
            # Aggregate episode results
            all_predictions.extend(episode_predictions)
            all_labels.extend(episode_labels)
            all_actions.extend(episode_actions)
            all_latencies.extend(episode_latencies)
            all_rewards.append(episode_reward)
            
            # Get episode statistics
            stats = self.env.get_episode_stats()
            episode_stats.append(stats)
            
            if verbose and (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f}")
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_actions = np.array(all_actions)
        all_latencies = np.array(all_latencies)
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            all_labels, all_predictions, all_actions, all_latencies, episode_stats
        )
        
        return metrics, {
            'predictions': all_predictions,
            'labels': all_labels,
            'actions': all_actions,
            'latencies': all_latencies,
            'rewards': all_rewards,
            'episode_stats': episode_stats
        }
    
    def _calculate_metrics(self, labels, predictions, actions, latencies, episode_stats):
        """Calculate performance metrics"""
        # Classification metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        # Latency metrics
        avg_latency = np.mean(latencies) * 1000  # Convert to milliseconds
        median_latency = np.median(latencies) * 1000
        p95_latency = np.percentile(latencies, 95) * 1000
        p99_latency = np.percentile(latencies, 99) * 1000
        
        # Action distribution
        action_distribution = {
            'allow': np.sum(actions == 0),
            'block': np.sum(actions == 1),
            'isolate': np.sum(actions == 2),
            'regulate': np.sum(actions == 3)
        }
        
        # Episode statistics
        avg_detection_rate = np.mean([s.get('detection_rate', 0) for s in episode_stats])
        avg_false_positive_rate = np.mean([s.get('false_positive_rate', 0) for s in episode_stats])
        avg_accuracy = np.mean([s.get('correct_actions', 0) / max(1, s.get('correct_actions', 0) + s.get('false_positives', 0) + s.get('false_negatives', 0)) for s in episode_stats])
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'avg_latency_ms': avg_latency,
            'median_latency_ms': median_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'action_distribution': action_distribution,
            'avg_detection_rate': avg_detection_rate,
            'avg_false_positive_rate': avg_false_positive_rate,
            'avg_episode_accuracy': avg_accuracy
        }
        
        return metrics
    
    def print_metrics(self, metrics):
        """Print evaluation metrics"""
        print("\n" + "=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
        
        print("\nClassification Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        
        print("\nConfusion Matrix:")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        print(f"  True Positives:  {metrics['true_positives']}")
        print("\n  Confusion Matrix:")
        print(f"    [{metrics['true_negatives']:6d}  {metrics['false_positives']:6d}]")
        print(f"    [{metrics['false_negatives']:6d}  {metrics['true_positives']:6d}]")
        
        print("\nResponse Latency:")
        print(f"  Average:  {metrics['avg_latency_ms']:.4f} ms")
        print(f"  Median:   {metrics['median_latency_ms']:.4f} ms")
        print(f"  95th percentile: {metrics['p95_latency_ms']:.4f} ms")
        print(f"  99th percentile: {metrics['p99_latency_ms']:.4f} ms")
        
        print("\nAction Distribution:")
        for action, count in metrics['action_distribution'].items():
            print(f"  {action.capitalize()}: {count}")
        
        print("\nEpisode Statistics:")
        print(f"  Average Detection Rate:     {metrics['avg_detection_rate']:.4f} ({metrics['avg_detection_rate']*100:.2f}%)")
        print(f"  Average False Positive Rate: {metrics['avg_false_positive_rate']:.4f} ({metrics['avg_false_positive_rate']*100:.2f}%)")
        print(f"  Average Episode Accuracy:   {metrics['avg_episode_accuracy']:.4f} ({metrics['avg_episode_accuracy']*100:.2f}%)")
        
        print("=" * 60)
    
    def plot_confusion_matrix(self, metrics, save_path='confusion_matrix.png'):
        """Plot confusion matrix"""
        cm = metrics['confusion_matrix']
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Benign', 'Attack'],
                    yticklabels=['Benign', 'Attack'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_action_distribution(self, metrics, save_path='action_distribution.png'):
        """Plot action distribution"""
        action_dist = metrics['action_distribution']
        actions = list(action_dist.keys())
        counts = list(action_dist.values())
        
        plt.figure(figsize=(8, 6))
        plt.bar(actions, counts, color=['green', 'red', 'orange', 'blue'])
        plt.title('Action Distribution')
        plt.xlabel('Action')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Action distribution saved to {save_path}")
        plt.close()


def main():
    """Main evaluation function"""
    print("=" * 60)
    print("RL-based IoT Intrusion Detection System - Evaluation")
    print("=" * 60)
    
    # Load preprocessor
    print("\n[Step 1] Loading preprocessor...")
    if not os.path.exists('preprocessor.pkl'):
        print("Error: Preprocessor file not found. Please run training first.")
        return
    
    preprocessor = DataPreprocessor(
        train_path='CICIOT23/train/train.csv',
        val_path='CICIOT23/validation/validation.csv',
        test_path='CICIOT23/test/test.csv'
    )
    preprocessor.load_preprocessor('preprocessor.pkl')
    
    # Load test data
    print("\n[Step 2] Loading test data...")
    # We need train data to get proper statistics for missing value imputation
    # But we'll only use test data for evaluation
    train_df, _, test_df = preprocessor.load_data(sample_size=None)
    
    # Preprocess test data using the same pipeline as training
    # We need to get feature columns and handle missing values consistently
    feature_cols = preprocessor.feature_columns
    X_test = test_df[feature_cols].copy()
    y_test = test_df['label'].copy()
    
    # Handle missing values using training statistics
    train_mean = getattr(preprocessor, 'train_mean', None)
    if train_mean is not None:
        # Use training mean for missing value imputation (consistent with training)
        X_test = X_test.fillna(train_mean)
    else:
        # Fallback to test mean if training mean not available
        X_test = X_test.fillna(X_test.mean())
    
    # Handle infinite values
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    if train_mean is not None:
        X_test = X_test.fillna(train_mean)
    else:
        X_test = X_test.fillna(X_test.mean())
    
    # Encode labels
    y_test_encoded = preprocessor.label_encoder.transform(y_test)
    
    # Normalize features using loaded scaler (trained on training data)
    X_test_scaled = preprocessor.scaler.transform(X_test)
    
    # Prepare data dictionary
    data = {
        'X_test': X_test_scaled.astype(np.float32),
        'y_test': y_test_encoded,
        'num_features': X_test_scaled.shape[1]
    }
    
    # Create environment with test data
    print("\n[Step 3] Creating evaluation environment...")
    # Get benign label index
    benign_label = preprocessor.label_encoder.transform(['BenignTraffic'])[0]
    print(f"Benign traffic label index: {benign_label}")
    
    env = IoTIntrusionDetectionEnv(
        X_data=data['X_test'],
        y_data=data['y_test'],
        max_steps=1000,
        benign_label_idx=benign_label
    )
    
    # Load trained agent
    print("\n[Step 4] Loading trained agent...")
    agent = DQNAgent(
        state_dim=data['num_features'],
        action_dim=4
    )
    
    model_path = 'models/dqn_agent_final.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first using train.py")
        return
    
    agent.load(model_path)
    agent.eval_mode()
    
    # Evaluate
    print("\n[Step 5] Evaluating agent...")
    evaluator = ModelEvaluator(agent, env, preprocessor.label_encoder)
    metrics, results = evaluator.evaluate(num_episodes=100, max_steps=1000, verbose=True)
    
    # Print metrics
    evaluator.print_metrics(metrics)
    
    # Save metrics
    print("\n[Step 6] Saving evaluation results...")
    os.makedirs('results', exist_ok=True)
    
    # Save metrics as JSON
    import json
    metrics_dict = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
    metrics_dict['confusion_matrix'] = metrics['confusion_matrix'].tolist()
    metrics_dict = _to_serializable(metrics_dict)
    
    with open('results/evaluation_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print("Evaluation metrics saved to results/evaluation_metrics.json")
    
    # Plot visualizations
    evaluator.plot_confusion_matrix(metrics, 'results/confusion_matrix.png')
    evaluator.plot_action_distribution(metrics, 'results/action_distribution.png')
    
    print("\n" + "=" * 60)
    print("Evaluation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

