# Reinforcement Learning for Adaptive Intrusion Detection and Response in IoT Networks

This project implements a Reinforcement Learning (RL) based system for adaptive intrusion detection and response in IoT networks using the CICIoT2023 dataset. The system uses a Deep Q-Network (DQN) agent that learns optimal strategies to detect and mitigate network attacks in real-time.

## Overview

The RL agent is trained to make intelligent decisions for network traffic flows, including:
- **Allow**: Permit normal/benign traffic
- **Block**: Block suspicious network flows
- **Isolate**: Isolate compromised IoT devices
- **Regulate**: Regulate network resources to minimize attack impact

The system continuously improves its decision-making through interaction with the network environment, adapting to new attack patterns.

## Dataset

The CICIoT2023 dataset contains traffic flows from IoT and PC devices under various attack scenarios:
- **Attack Types**: DDoS, DoS, ransomware, spoofing, brute-force, web attacks, and more (34 total attack types)
- **Features**: 46 network traffic features (flow duration, protocol types, packet statistics, etc.)
- **Dataset Size**: ~5.5M training samples, ~1.2M validation samples, ~1.2M test samples

## Project Structure

```
.
├── preprocessing.py          # Data preprocessing and feature engineering
├── rl_environment.py         # RL environment for network traffic simulation
├── dqn_agent.py              # DQN agent implementation
├── train.py                  # Training script
├── evaluate.py               # Evaluation script with metrics
├── dataset_analysis.py       # Dataset exploration and statistics script
├── visualize.py              # Visualization scripts
├── main.py                   # Main execution script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── CICIOT23/                 # Dataset directory
│   ├── train/
│   ├── validation/
│   └── test/
├── models/                   # Saved model files (created during training)
└── results/                  # Evaluation results and visualizations (created during execution)
```

## Installation

### Recommended: Using Virtual Environment

It is highly recommended to use a virtual environment to isolate project dependencies:

#### On Windows:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

#### On Linux/Mac:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

#### Deactivate virtual environment:
```bash
deactivate
```

### Alternative: Global Installation

If you prefer not to use a virtual environment (not recommended):

1. **Clone or download this repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

**Note**: Using a virtual environment is recommended to avoid conflicts with other Python projects and to maintain a clean development environment.

3. **Ensure the CICIoT2023 dataset is available**:
   - The dataset should be in the `CICIOT23/` directory
   - Training data: `CICIOT23/train/train.csv`
   - Validation data: `CICIOT23/validation/validation.csv`
   - Test data: `CICIOT23/test/test.csv`
   
   **Note**: The dataset files are large and are typically excluded from git via `.gitignore`. Make sure to download and place the dataset files in the correct directory structure before running the scripts.

## Usage

### Quick Start

Run the complete pipeline (training, evaluation, and visualization):
```bash
python main.py all
```

### Individual Steps

#### 0. Dataset Analysis (optional but recommended)

Generate descriptive statistics, label distributions, correlation heatmaps, and feature distribution plots:
```bash
python dataset_analysis.py --output-dir results/dataset_analysis --sample-size 100000
```

Key options:
- `--sample-size`: Number of rows to sample from the combined dataset (default 100,000). Use `--no-sample` to process the full dataset (memory intensive).
- `--correlation-sample-size`: Rows used when computing the correlation matrix (default 10,000).
- `--max-corr-features`: Number of high-variance features to include in the correlation heatmap (default 30).

Outputs are saved in the specified `--output-dir` and include:
- `dataset_metadata.json`
- `label_distribution.csv` and per-split counts
- `feature_summary.csv`, `missing_values.csv`, `top_variance_features.csv`
- `correlation_matrix.csv` and `correlation_heatmap.png`
- `feature_distributions.png`
- `label_distribution_top20.png`

#### 1. Training

Train the RL agent:
```bash
python train.py
```

Or use the main script:
```bash
python main.py train
```

**Configuration options** (edit `train.py`):
- `sample_size`: Sample size for faster development (None for full dataset)
- `num_episodes`: Number of training episodes (default: 500)
- `max_steps_per_episode`: Maximum steps per episode (default: 500)
- `lr`: Learning rate (default: 0.001)
- `gamma`: Discount factor (default: 0.99)
- `epsilon_start/end/decay`: Epsilon-greedy exploration parameters

**Outputs**:
- Trained models saved in `models/` directory
- Training history saved as `training_history.json`
- Preprocessor saved as `preprocessor.pkl`

#### 2. Evaluation

Evaluate the trained model:
```bash
python evaluate.py
```

Or use the main script:
```bash
python main.py evaluate
```

**Metrics computed**:
- Accuracy
- Precision
- Recall
- F1-Score
- Response Latency (average, median, 95th, 99th percentiles)
- Confusion Matrix
- Action Distribution

**Outputs**:
- Evaluation metrics saved in `results/evaluation_metrics.json`
- Confusion matrix plot: `results/confusion_matrix.png`
- Action distribution plot: `results/action_distribution.png`

#### 3. Visualization

Generate training performance visualizations:
```bash
python visualize.py
```

Or use the main script:
```bash
python main.py visualize
```

**Generated visualizations**:
- Training rewards over episodes
- Training loss over episodes
- Detection metrics (detection rate, false positive rate)
- Accuracy over time
- Learning curves summary
- Agent adaptation analysis

**Outputs**: All visualizations saved in `results/` directory

## System Architecture

### RL Environment

The `IoTIntrusionDetectionEnv` class implements a Gymnasium-compatible RL environment:
- **State Space**: 46-dimensional feature vector from network traffic
- **Action Space**: 4 discrete actions (Allow, Block, Isolate, Regulate)
- **Reward Function**: 
  - Positive rewards for correct detections and appropriate responses
  - Negative rewards (penalties) for false positives and false negatives
  - Higher rewards for more effective responses (isolate > block > regulate)

### DQN Agent

The DQN agent implements:
- **Deep Q-Network**: Multi-layer neural network for Q-value estimation
- **Experience Replay**: Stores and samples past experiences for stable training
- **Target Network**: Separate target network for stable Q-learning updates
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation

### Training Process

1. **Data Preprocessing**:
   - Load and normalize features
   - Encode labels
   - Handle missing values

2. **Environment Setup**:
   - Create RL environment with training data
   - Configure reward function

3. **Agent Training**:
   - Agent interacts with environment
   - Collects experiences (state, action, reward, next_state)
   - Updates Q-network using experience replay
   - Gradually reduces exploration (epsilon decay)

4. **Evaluation**:
   - Test agent on held-out test set
   - Compute performance metrics
   - Analyze agent behavior

## Performance Metrics

The system evaluates the following metrics:

### Classification Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of predicted attacks that are actual attacks
- **Recall**: Proportion of actual attacks that are detected
- **F1-Score**: Harmonic mean of precision and recall

### Response Metrics
- **Detection Rate**: Percentage of attacks successfully detected
- **False Positive Rate**: Percentage of benign traffic incorrectly flagged
- **Response Latency**: Time taken to make decisions (milliseconds)

### Action Analysis
- Distribution of actions taken (Allow, Block, Isolate, Regulate)
- Effectiveness of different response strategies

## Results Analysis

The visualization module provides insights into:
1. **Training Progress**: How the agent improves over episodes
2. **Learning Curves**: Rewards, accuracy, and detection metrics over time
3. **Agent Adaptation**: How the agent adapts to different attack patterns across training phases
4. **Performance Trade-offs**: Balance between detection rate and false positive rate

## Customization

### Reward Function

Modify reward parameters in `rl_environment.py`:
```python
reward_config = {
    'correct_detection': 10.0,
    'correct_benign': 2.0,
    'false_positive': -5.0,
    'false_negative': -20.0,
    # ... more parameters
}
```

### Network Architecture

Modify DQN architecture in `dqn_agent.py`:
```python
agent = DQNAgent(
    state_dim=46,
    action_dim=4,
    hidden_dims=[128, 128, 64]  # Customize hidden layers
)
```

### Training Parameters

Adjust training hyperparameters in `train.py`:
- Learning rate, discount factor, epsilon parameters
- Batch size, replay buffer size
- Number of episodes, steps per episode

## Requirements

- Python 3.8+
- PyTorch 2.1+
- NumPy, Pandas
- Scikit-learn
- Gymnasium (for RL environment)
- Matplotlib, Seaborn (for visualization)

See `requirements.txt` for complete list.

## Troubleshooting

### Memory Issues
If you encounter memory issues with the full dataset:
- Set `sample_size` in `train.py` to a smaller value (e.g., 100000)
- Reduce `max_steps_per_episode`
- Reduce `batch_size` or `memory_size` in agent configuration

### Training is Slow
- Use GPU if available (PyTorch will automatically use CUDA)
- Reduce number of episodes for initial testing
- Use dataset sampling for faster iteration

### Model Not Found
- Ensure you've trained the model first using `train.py`
- Check that `models/dqn_agent_final.pth` exists
- Specify correct model path in `evaluate.py`

## Git Setup

This project includes a `.gitignore` file that excludes:
- Virtual environment directories (`venv/`, `env/`)
- Model files (`.pth`, `.pkl`)
- Training results and visualizations (`results/`, `models/`)
- Python cache files (`__pycache__/`, `*.pyc`)
- IDE-specific files (`.vscode/`, `.idea/`)
- Large dataset files (commented out - uncomment if needed)
- Log files and temporary files
