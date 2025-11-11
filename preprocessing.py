"""
Data preprocessing module for CICIoT2023 dataset
Handles feature extraction, normalization, and label encoding
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os


class DataPreprocessor:
    """Preprocesses CICIoT2023 dataset for RL training"""
    
    def __init__(self, train_path, val_path, test_path):
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
    def load_data(self, sample_size=None):
        """Load dataset with optional sampling for faster processing"""
        print("Loading training data...")
        train_df = pd.read_csv(self.train_path)
        
        print("Loading validation data...")
        val_df = pd.read_csv(self.val_path)
        
        print("Loading test data...")
        test_df = pd.read_csv(self.test_path)
        
        # Sample data if specified (for faster development/testing)
        if sample_size:
            print(f"Sampling {sample_size} rows from each dataset...")
            train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
            val_df = val_df.sample(n=min(sample_size, len(val_df)), random_state=42)
            test_df = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)
        
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")
        print(f"Test samples: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def preprocess(self, train_df, val_df, test_df, handle_missing='mean'):
        """Preprocess the datasets"""
        print("\nPreprocessing data...")
        
        # Separate features and labels
        feature_cols = [col for col in train_df.columns if col != 'label']
        self.feature_columns = feature_cols
        
        X_train = train_df[feature_cols].copy()
        y_train = train_df['label'].copy()
        X_val = val_df[feature_cols].copy()
        y_val = val_df['label'].copy()
        X_test = test_df[feature_cols].copy()
        y_test = test_df['label'].copy()
        
        # Handle missing values
        if handle_missing == 'mean':
            X_train = X_train.fillna(X_train.mean())
            X_val = X_val.fillna(X_train.mean())
            X_test = X_test.fillna(X_train.mean())
        elif handle_missing == 'drop':
            X_train = X_train.dropna()
            X_val = X_val.dropna()
            X_test = X_test.dropna()
        
        # Handle infinite values
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.mean())
        X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(X_train.mean())
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.mean())
        
        # Encode labels
        print("Encoding labels...")
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Normalize features
        print("Normalizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get label mapping
        label_mapping = dict(zip(self.label_encoder.classes_, 
                                range(len(self.label_encoder.classes_))))
        
        print(f"\nNumber of classes: {len(label_mapping)}")
        print(f"Feature dimensions: {X_train_scaled.shape[1]}")
        print(f"Label distribution (train):")
        unique, counts = np.unique(y_train_encoded, return_counts=True)
        for label_idx, count in zip(unique[:10], counts[:10]):
            label_name = self.label_encoder.inverse_transform([label_idx])[0]
            print(f"  {label_name}: {count}")
        if len(unique) > 10:
            print(f"  ... and {len(unique) - 10} more classes")
        
        return {
            'X_train': X_train_scaled.astype(np.float32),
            'y_train': y_train_encoded,
            'X_val': X_val_scaled.astype(np.float32),
            'y_val': y_val_encoded,
            'X_test': X_test_scaled.astype(np.float32),
            'y_test': y_test_encoded,
            'label_mapping': label_mapping,
            'num_classes': len(label_mapping),
            'num_features': X_train_scaled.shape[1]
        }
    
    def save_preprocessor(self, filepath='preprocessor.pkl', train_mean=None):
        """Save preprocessor objects"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns,
                'train_mean': train_mean  # Save training mean for missing value imputation
            }, f)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath='preprocessor.pkl'):
        """Load preprocessor objects"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.label_encoder = data['label_encoder']
            self.feature_columns = data['feature_columns']
            self.train_mean = data.get('train_mean', None)  # Load training mean if available
        print(f"Preprocessor loaded from {filepath}")


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = DataPreprocessor(
        train_path='CICIOT23/train/train.csv',
        val_path='CICIOT23/validation/validation.csv',
        test_path='CICIOT23/test/test.csv'
    )
    
    # Load with sampling for testing (remove sample_size for full dataset)
    train_df, val_df, test_df = preprocessor.load_data(sample_size=10000)
    
    # Preprocess
    data = preprocessor.preprocess(train_df, val_df, test_df)
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    print("\nPreprocessing completed successfully!")

