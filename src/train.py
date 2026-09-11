"""
Training pipeline for enzyme function prediction.

Workflow:
  1. Load pre-extracted ESM-2 embeddings + labels
  2. Split into train/val/test (stratified)
  3. Train sklearn baselines (LR, RF, GB)
  4. Train MLP classifier (PyTorch)
  5. Evaluate on test set
  6. Save results + best model
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import MLPClassifier, ResidualMLP, train_sklearn_baselines

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
EC_CLASSES = [f"EC{i}" for i in range(1, 7)]


def load_data():
    """Load embeddings and labels, create stratified splits."""
    embeddings_path = DATA_DIR / "esm2_embeddings.npy"
    csv_path = DATA_DIR / "enzymes_swissprot.csv"

    if not embeddings_path.exists():
        print(f"Error: {embeddings_path} not found. Run src/embeddings.py first.")
        sys.exit(1)

    embeddings = np.load(embeddings_path)
    df = pd.read_csv(csv_path)

    # Labels: ec_class is 1-6, convert to 0-5 for PyTorch
    labels = (df["ec_class"].values - 1).astype(np.int64)

    print(f"Loaded {len(embeddings)} samples, embedding dim: {embeddings.shape[1]}")
    print(f"Class distribution: {np.bincount(labels)}")

    # Stratified split: 70% train, 15% val, 15% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        embeddings, labels, test_size=0.15, stratify=labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.176,  # 0.176 of 85% ≈ 15% of total
        stratify=y_trainval,
        random_state=42,
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val:   {len(X_val)}")
    print(f"  Test:  {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_pytorch_model(X_train, y_train, X_val, y_val,
                         model_class="mlp", epochs=50, batch_size=64,
                         lr=1e-3, device="cuda"):
    """Train a PyTorch classifier on ESM-2 embeddings."""
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    if model_class == "mlp":
        model = MLPClassifier(input_dim=input_dim, num_classes=num_classes)
    elif model_class == "residual":
        model = ResidualMLP(input_dim=input_dim, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_class}")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(y_val)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_val_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
    save_path = RESULTS_DIR / f"best_{model_class}.pt"

    print(f"\nTraining {model_class.upper()} for {epochs} epochs...")
    print(f"{'='*50}")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        val_acc = accuracy_score(all_labels, all_preds)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "epoch": epoch,
            }, save_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val F1: {val_f1:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

    print(f"\nBest Val F1: {best_val_f1:.4f}")
    print(f"Model saved to: {save_path}")

    return model, history


def evaluate_test_set(model_class="mlp", X_test=None, y_test=None, device="cuda"):
    """Load best model and evaluate on test set."""
    input_dim = X_test.shape[1]
    num_classes = len(np.unique(y_test))

    if model_class == "mlp":
        model = MLPClassifier(input_dim=input_dim, num_classes=num_classes)
    elif model_class == "residual":
        model = ResidualMLP(input_dim=input_dim, num_classes=num_classes)

    checkpoint_path = RESULTS_DIR / f"best_{model_class}.pt"
    if not checkpoint_path.exists():
        print(f"Error: {checkpoint_path} not found.")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        outputs = model(X_tensor)
        preds = outputs.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    report = classification_report(y_test, preds, target_names=EC_CLASSES, output_dict=True)

    print(f"\n{'='*50}")
    print(f"TEST SET RESULTS — {model_class.upper()}")
    print(f"{'='*50}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(classification_report(y_test, preds, target_names=EC_CLASSES))

    return {"accuracy": acc, "f1_macro": f1, "report": report, "predictions": preds}


def save_all_results(sklearn_results, pytorch_results, history):
    """Save all experiment results to JSON."""
    output = {
        "sklearn_baselines": {},
        "pytorch_models": {},
        "training_history": {},
    }

    for name, res in sklearn_results.items():
        output["sklearn_baselines"][name] = {
            "accuracy": float(res["accuracy"]),
            "f1_macro": float(res["f1_macro"]),
        }

    for name, res in pytorch_results.items():
        if res is not None:
            output["pytorch_models"][name] = {
                "accuracy": float(res["accuracy"]),
                "f1_macro": float(res["f1_macro"]),
            }

    for name, h in history.items():
        output["training_history"][name] = {
            k: [float(v) for v in vals] for k, vals in h.items()
        }

    output_path = RESULTS_DIR / "experiment_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    """Full training pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # 2. Sklearn baselines
    print("\n" + "="*60)
    print("SKLEARN BASELINES")
    print("="*60)
    sklearn_results = train_sklearn_baselines(X_train, y_train, X_test, y_test)

    # 3. PyTorch models
    pytorch_results = {}
    histories = {}

    for model_class in ["mlp", "residual"]:
        model, history = train_pytorch_model(
            X_train, y_train, X_val, y_val,
            model_class=model_class,
            epochs=50,
            batch_size=64,
            lr=1e-3,
            device=device,
        )
        histories[model_class] = history
        pytorch_results[model_class] = evaluate_test_set(
            model_class, X_test, y_test, device
        )

    # 4. Save everything
    save_all_results(sklearn_results, pytorch_results, histories)

    # 5. Print summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    all_results = {}
    for name, res in sklearn_results.items():
        all_results[name] = {"acc": res["accuracy"], "f1": res["f1_macro"]}
    for name, res in pytorch_results.items():
        if res:
            all_results[name.upper()] = {"acc": res["accuracy"], "f1": res["f1_macro"]}

    for name, metrics in sorted(all_results.items(), key=lambda x: x[1]["f1"], reverse=True):
        print(f"  {name:25s} | Acc: {metrics['acc']:.4f} | F1: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
