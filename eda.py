"""
Exploratory Data Analysis script.
Run this once to generate summary stats and plots.
Outputs go to static/eda/ for reference.
Usage: python eda.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config
from feature_engineering import add_engineered_features


def load_data():
    df = pd.read_csv(config.DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn types:\n{df.dtypes}\n")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}\n")
    print(f"Target stats:\n{df[config.TARGET_COLUMN].describe()}\n")
    return df


def plot_target_distribution(df, output_dir):
    """Histogram of the wellbeing score."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[config.TARGET_COLUMN], bins=30, color="#4a90d9", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Mental Wellbeing Score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Wellbeing Scores")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "target_distribution.png"), dpi=150)
    plt.close(fig)
    print("Saved: target_distribution.png")


def plot_correlation_heatmap(df, output_dir):
    """Heatmap of feature correlations with the target."""
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax,
                linewidths=0.5, square=True)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=150)
    plt.close(fig)
    print("Saved: correlation_heatmap.png")


def plot_top_correlations(df, output_dir):
    """Bar chart of features most correlated with the target."""
    numeric_df = df.select_dtypes(include=["number"])
    target_corr = numeric_df.corr()[config.TARGET_COLUMN].drop(config.TARGET_COLUMN)
    top = target_corr.abs().sort_values(ascending=False).head(15)
    values = target_corr[top.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#e74c3c" if v < 0 else "#27ae60" for v in values]
    ax.barh(values.index[::-1], values.values[::-1], color=colors[::-1])
    ax.set_xlabel("Correlation with Wellbeing Score")
    ax.set_title("Top 15 Feature Correlations")
    ax.axvline(x=0, color="gray", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "top_correlations.png"), dpi=150)
    plt.close(fig)
    print("Saved: top_correlations.png")


def plot_engineered_features(df, output_dir):
    """Distribution of the five engineered composite scores."""
    df = add_engineered_features(df)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#1abc9c"]

    for i, feat in enumerate(config.ENGINEERED_FEATURES):
        axes[i].hist(df[feat], bins=25, color=colors[i], edgecolor="white", alpha=0.85)
        label = feat.replace("_score", "").replace("_", " ").title()
        axes[i].set_title(label)
        axes[i].set_xlabel("Score")

    plt.suptitle("Engineered Feature Distributions", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "engineered_features.png"), dpi=150)
    plt.close(fig)
    print("Saved: engineered_features.png")


def main():
    output_dir = os.path.join(config.BASE_DIR, "static", "eda")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("Exploratory Data Analysis")
    print("=" * 50)

    df = load_data()
    plot_target_distribution(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_top_correlations(df, output_dir)
    plot_engineered_features(df, output_dir)

    print("\nEDA complete! Plots saved to static/eda/")


if __name__ == "__main__":
    main()
