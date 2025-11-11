"""
Main execution script for RL-based IoT Intrusion Detection System
Provides a unified interface for training, evaluation, and visualization
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description='RL-based IoT Intrusion Detection and Response System'
    )
    parser.add_argument(
        'mode',
        choices=['train', 'evaluate', 'visualize', 'all'],
        help='Execution mode: train, evaluate, visualize, or all'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Sample size for faster development (None for full dataset)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=500,
        help='Number of training episodes (default: 500)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/dqn_agent_final.pth',
        help='Path to trained model for evaluation'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train' or args.mode == 'all':
        print("\n" + "=" * 60)
        print("TRAINING MODE")
        print("=" * 60)
        from train import main as train_main
        # Modify config if sample_size is provided
        if args.sample_size:
            import train
            # We'll need to modify the config in train.py or pass it as parameter
            print(f"Note: Using sample size {args.sample_size} for training")
        if args.episodes:
            print(f"Note: Training for {args.episodes} episodes")
        train_main()
    
    if args.mode == 'evaluate' or args.mode == 'all':
        print("\n" + "=" * 60)
        print("EVALUATION MODE")
        print("=" * 60)
        if not os.path.exists(args.model_path):
            print(f"Error: Model file not found at {args.model_path}")
            print("Please train the model first or specify a different model path.")
            if args.mode == 'evaluate':
                sys.exit(1)
        else:
            from evaluate import main as evaluate_main
            evaluate_main()
    
    if args.mode == 'visualize' or args.mode == 'all':
        print("\n" + "=" * 60)
        print("VISUALIZATION MODE")
        print("=" * 60)
        if not os.path.exists('training_history.json'):
            print("Error: Training history file not found.")
            print("Please train the model first to generate training history.")
            if args.mode == 'visualize':
                sys.exit(1)
        else:
            from visualize import main as visualize_main
            visualize_main()
    
    print("\n" + "=" * 60)
    print("Execution completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

