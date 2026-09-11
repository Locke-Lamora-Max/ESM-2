"""
Visualization module for enzyme function prediction experiments.

Generates:
  1. Training curves (loss + F1)
  2. Confusion matrix heatmap
  3. Model comparison bar chart
  4. Per-class F1 comparison
  5. Sequence length distribution
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_DIR = Path(__file__).parent.parent / "data"

EC_CLASSES = [f"EC{i}" for i in range(1, 7)]
EC_NAMES = [
    "Oxidoreductases", "Transferases", "Hydrolases",
    "Lyases", "Isomerases", "Ligases"
]

# Set style
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox_inches": "tight",
})


def plot_training_curves(history, save=True):
    """Plot training loss, validation loss, and validation F1."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#2196F3", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", color="#E91E63", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1 curve
    axes[1].plot(epochs, history["val_f1"], label="Val Macro F1", color="#4CAF50", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy", color="#FF9800",
                 linewidth=2, linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation Performance")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("ESM-2 + MLP — Training Dynamics", fontsize=15, y=1.02)
    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / "training_curves.png")
        print(f"Saved: {FIGURES_DIR / 'training_curves.png'}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save=True):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Absolute counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=EC_CLASSES, yticklabels=EC_CLASSES, ax=axes[0])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Confusion Matrix (Counts)")

    # Percentages
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=EC_CLASSES, yticklabels=EC_CLASSES, ax=axes[1])
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Confusion Matrix (%)")

    plt.suptitle("Enzyme EC Classification — ESM-2 + MLP", fontsize=15, y=1.02)
    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / "confusion_matrix.png")
        print(f"Saved: {FIGURES_DIR / 'confusion_matrix.png'}")
    plt.show()


def plot_model_comparison(results_path=None, save=True):
    """Bar chart comparing all models."""
    if results_path is None:
        results_path = RESULTS_DIR / "experiment_results.json"

    with open(results_path) as f:
        results = json.load(f)

    model_names = []
    f1_scores = []
    accuracies = []

    for name, res in results.get("sklearn_baselines", {}).items():
        model_names.append(name)
        f1_scores.append(res["f1_macro"])
        accuracies.append(res["accuracy"])

    for name, res in results.get("pytorch_models", {}).items():
        model_names.append(f"ESM-2 + {name.upper()}")
        f1_scores.append(res["f1_macro"])
        accuracies.append(res["accuracy"])

    # Sort by F1
    sorted_idx = np.argsort(f1_scores)[::-1]
    model_names = [model_names[i] for i in sorted_idx]
    f1_scores = [f1_scores[i] for i in sorted_idx]
    accuracies = [accuracies[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(model_names)))
    x = np.arange(len(model_names))

    bars = ax.bar(x, f1_scores, color=colors, width=0.6, label="Macro F1")
    ax.bar(x, accuracies, color=colors, width=0.3, alpha=0.5, label="Accuracy")

    for i, (f1, acc) in enumerate(zip(f1_scores, accuracies)):
        ax.text(i, f1 + 0.01, f"{f1:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Enzyme EC Classification (ESM-2 Embeddings)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_DIR / "model_comparison.png")
        print(f"Saved: {FIGURES_DIR / 'model_comparison.png'}")
    plt.show()


def plot_class_distribution(save=True):
    """Plot distribution of EC classes in dataset."""
    csv_path = DATA_DIR / "enzymes_swissprot.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Class counts
    counts = df["ec_class"].value_counts().sort_index()
    colors = plt.cm.Set2(np.linspace(0, 1, 6))
    axes[0].bar(EC_CLASSES, counts.values, color=colors)
    axes[0].set_xlabel("EC Class")
    axes[0].set_ylabel("Number of Sequences")
    axes[0].set_title("Dataset: EC Class Distribution")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 20, str(v), ha="center", fontweight="bold")

    # Sequence length distribution
    axes[1].hist(df["length"], bins=50, color="#2196F3", edgecolor="white", alpha=0.8)
    axes[1].axvline(df["length"].median(), color="red", linestyle="--",
                    label=f'Median: {df["length"].median():.0f}')
    axes[1].set_xlabel("Sequence Length (amino acids)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Sequence Length Distribution")
    axes[1].legend()

    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_DIR / "dataset_overview.png")
        print(f"Saved: {FIGURES_DIR / 'dataset_overview.png'}")
    plt.show()


def plot_per_class_f1(results_path=None, save=True):
    """Per-class F1 comparison across models."""
    if results_path is None:
        results_path = RESULTS_DIR / "experiment_results.json"

    with open(results_path) as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    model_data = {}
    for name, res in results.get("sklearn_baselines", {}).items():
        model_data[name] = [res["report"][c]["f1-score"] for c in EC_CLASSES]
    for name, res in results.get("pytorch_models", {}).items():
        model_data[f"ESM-2 + {name.upper()}"] = [res["report"][c]["f1-score"] for c in EC_CLASSES]

    x = np.arange(len(EC_CLASSES))
    width = 0.8 / len(model_data)
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_data)))

    for i, (name, scores) in enumerate(model_data.items()):
        offset = (i - len(model_data)/2 + 0.5) * width
        bars = ax.bar(x + offset, scores, width, label=name, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{n}" for c, n in zip(EC_CLASSES, EC_NAMES)],
                       fontsize=9, ha="center")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score Comparison")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(FIGURES_DIR / "per_class_f1.png")
        print(f"Saved: {FIGURES_DIR / 'per_class_f1.png'}")
    plt.show()


def generate_all_figures():
    """Generate all publication-quality figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating figures...\n")

    # 1. Dataset overview
    plot_class_distribution()

    # 2. Model comparison
    plot_model_comparison()

    # 3. Per-class F1
    plot_per_class_f1()

    # 4. Training curves (if history exists)
    results_path = RESULTS_DIR / "experiment_results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        history = results.get("training_history", {}).get("mlp")
        if history:
            plot_training_curves(history)

    print("\nAll figures generated in:", FIGURES_DIR)


if __name__ == "__main__":
    generate_all_figures()
