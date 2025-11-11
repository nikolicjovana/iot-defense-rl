"""Dataset analysis utilities for the CICIoT2023 dataset.

This script aggregates the train/validation/test splits, computes descriptive
statistics, label distributions, correlation matrices, and selected feature
visualisations to help understand the data prior to model training.

Usage
-----
python dataset_analysis.py --output-dir results/dataset_analysis --sample-size 50000
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATASET_DEFAULT_PATHS = {
    "train": "CICIOT23/train/train.csv",
    "validation": "CICIOT23/validation/validation.csv",
    "test": "CICIOT23/test/test.csv",
}

RANDOM_STATE = 42


def load_dataset(paths: Dict[str, str], sample_size: Optional[int] = None) -> pd.DataFrame:
    """Load dataset splits and optionally down-sample for quicker analysis."""
    frames: List[pd.DataFrame] = []
    for split_name, file_path in paths.items():
        if not Path(file_path).exists():
            print(f"Warning: {split_name} file not found at {file_path}. Skipping.")
            continue
        print(f"Loading {split_name} split from {file_path} ...")
        frame = pd.read_csv(file_path)
        frame["split"] = split_name
        frames.append(frame)

    if not frames:
        raise FileNotFoundError("No dataset files were loaded. Please check the paths.")

    full_df = pd.concat(frames, ignore_index=True)
    print(f"Total rows loaded: {len(full_df):,}")

    if sample_size is not None and sample_size < len(full_df):
        print(f"Sampling {sample_size:,} rows for analysis ...")
        full_df = full_df.sample(n=sample_size, random_state=RANDOM_STATE)

    return full_df.reset_index(drop=True)


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """Persist a dataframe to CSV with basic logging."""
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def analyse_labels(df: pd.DataFrame, output_dir: Path) -> None:
    """Compute label distribution overall and per split."""
    label_counts = (
        df.groupby("label")["label"].count().rename("count").sort_values(ascending=False)
    )
    save_dataframe(label_counts.reset_index(), output_dir / "label_distribution.csv")

    split_label_counts = (
        df.groupby(["split", "label"]).size().reset_index(name="count")
    )
    save_dataframe(split_label_counts, output_dir / "label_distribution_by_split.csv")

    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=label_counts.reset_index().head(20),
        x="label",
        y="count",
        palette="viridis",
    )
    plt.xticks(rotation=70, ha="right")
    plt.title("Top 20 attack types by frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "label_distribution_top20.png", dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'label_distribution_top20.png'}")


def analyse_features(df: pd.DataFrame, output_dir: Path, top_n: int = 20) -> None:
    """Compute descriptive statistics and missing value report for numeric features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numeric feature count: {len(numeric_cols)}")

    summary = df[numeric_cols].describe().transpose().reset_index().rename(columns={"index": "feature"})
    save_dataframe(summary, output_dir / "feature_summary.csv")

    missing_values = (
        df[numeric_cols]
        .isna()
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature", 0: "missing_count"})
    )
    save_dataframe(missing_values, output_dir / "missing_values.csv")

    # Feature variance ranking to identify most informative features
    variances = df[numeric_cols].var().sort_values(ascending=False)
    top_features = variances.head(top_n).index.tolist()
    save_dataframe(
        variances.head(top_n).reset_index().rename(columns={"index": "feature", 0: "variance"}),
        output_dir / "top_variance_features.csv",
    )

    # Plot distributions for top variance features (limited to avoid clutter)
    bins = 50
    max_plots = min(6, len(top_features))
    if max_plots > 0:
        fig, axes = plt.subplots(max_plots, 1, figsize=(10, 4 * max_plots))
        if max_plots == 1:
            axes = [axes]
        for ax, feature in zip(axes, top_features[:max_plots]):
            sns.histplot(data=df, x=feature, hue="split", bins=bins, ax=ax, element="step", stat="density")
            ax.set_title(f"Distribution of {feature}")
        plt.tight_layout()
        plt.savefig(output_dir / "feature_distributions.png", dpi=300)
        plt.close()
        print(f"Saved: {output_dir / 'feature_distributions.png'}")


def analyse_correlations(
    df: pd.DataFrame,
    output_dir: Path,
    correlation_sample_size: Optional[int] = 10000,
    max_features: int = 30,
) -> None:
    """Compute correlation matrix and save heatmap for top-variance features."""
    numeric_df = df.select_dtypes(include=[np.number])
    if correlation_sample_size and len(numeric_df) > correlation_sample_size:
        numeric_df = numeric_df.sample(n=correlation_sample_size, random_state=RANDOM_STATE)
        print(f"Correlation analysis on a sample of {len(numeric_df):,} rows")
    else:
        print(f"Correlation analysis on {len(numeric_df):,} rows")

    variances = numeric_df.var().sort_values(ascending=False)
    selected_features = variances.head(max_features).index.tolist()
    print(f"Computing correlations for top {len(selected_features)} features")

    corr_matrix = numeric_df[selected_features].corr()
    save_dataframe(corr_matrix.reset_index(), output_dir / "correlation_matrix.csv")

    plt.figure(figsize=(min(20, 0.6 * len(selected_features)), min(16, 0.6 * len(selected_features))))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Feature Correlation Heatmap (top variance features)")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'correlation_heatmap.png'}")


def aggregate_metadata(df: pd.DataFrame, output_dir: Path) -> None:
    """Persist simple metadata overview as JSON."""
    metadata = {
        "total_rows": int(len(df)),
        "columns": df.columns.tolist(),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "split_distribution": df["split"].value_counts().to_dict(),
    }

    with open(output_dir / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {output_dir / 'dataset_metadata.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CICIoT2023 dataset analysis")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/dataset_analysis",
        help="Directory to store analysis outputs",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100_000,
        help="Number of rows to sample for analysis (None for all rows)",
    )
    parser.add_argument(
        "--correlation-sample-size",
        type=int,
        default=10_000,
        help="Rows to sample when computing correlation matrix (None for all rows)",
    )
    parser.add_argument(
        "--max-corr-features",
        type=int,
        default=30,
        help="Maximum number of features to include in correlation heatmap",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling and use the entire dataset (may require significant memory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    sample_size = None if args.no_sample else args.sample_size

    df = load_dataset(DATASET_DEFAULT_PATHS, sample_size=sample_size)

    aggregate_metadata(df, output_dir)
    analyse_labels(df, output_dir)
    analyse_features(df, output_dir)
    analyse_correlations(
        df,
        output_dir,
        correlation_sample_size=args.correlation_sample_size,
        max_features=args.max_corr_features,
    )

    print("\nDataset analysis completed. Outputs saved to:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
