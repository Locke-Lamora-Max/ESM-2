"""
Training pipeline for enzyme function prediction.

Workflow:
  1. Load pre-extracted ESM-2 embeddings + labels
  2. Split into train/val/test (stratified)
  3. Train sklearn baselines (LR, RF, GB)
  4. Train MLP classifier (PyTorch)
  5. Train Hierarchical EC classifier (level-by-level)
  6. Train Multi-task EC+GO model (auxiliary GO head)
  7. Evaluate on test set
  8. Save results + best model
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
from src.models import (
    MLPClassifier, ResidualMLP, HierarchicalECClassifier, MultiTaskECGO,
    train_sklearn_baselines, EC_CLASSES, EC_SUBCLASS_COUNTS, TOP_GO_TERMS,
)

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


def load_hierarchical_labels(csv_path):
    """
    Load hierarchical EC labels from the CSV.

    The CSV contains ec_number strings like "2.7.1.1".
    We parse them into:
      - level1: main class (0-5, i.e., EC 1-6 → 0-5)
      - level2: subclass index (0-indexed within the main class)
    """
    df = pd.read_csv(csv_path)
    level1_labels = (df["ec_class"].values - 1).astype(np.int64)

    level2_labels = []
    for ec_str, ec_class in zip(df["ec_number"].values, df["ec_class"].values):
        parts = str(ec_str).split(".")
        try:
            subclass = int(parts[1]) - 1  # 0-indexed
        except (IndexError, ValueError):
            subclass = 0
        # Clamp to the subclass head's width so CrossEntropyLoss can
        # never hit an out-of-range index (which crashes the CUDA kernel)
        max_subclass = EC_SUBCLASS_COUNTS.get(int(ec_class), 27) - 1
        level2_labels.append(min(max(0, subclass), max_subclass))
    level2_labels = np.array(level2_labels, dtype=np.int64)

    return level1_labels, level2_labels


def load_go_labels(csv_path):
    """
    Load GO annotation labels from the CSV.

    Expects a 'go_terms' column with semicolon-separated GO IDs.
    Returns binary matrix of shape (n_samples, len(TOP_GO_TERMS)).
    """
    df = pd.read_csv(csv_path)
    go_col = "go_terms"

    if go_col not in df.columns:
        print(f"\n  WARNING: '{go_col}' column not found in CSV.")
        print(f"  GO annotation labels require re-running data download")
        print(f"  (python scripts/run_pipeline.py --download) to fetch GO data.\n")
        print(f"  Skipping the GO auxiliary task and training with the EC task only.")
        return np.zeros((len(df), len(TOP_GO_TERMS)), dtype=np.float32)

    go_labels = np.zeros((len(df), len(TOP_GO_TERMS)), dtype=np.float32)
    term_to_idx = {t: i for i, t in enumerate(TOP_GO_TERMS)}

    for i, terms_str in enumerate(df[go_col].values):
        if pd.isna(terms_str):
            continue
        for term in str(terms_str).split(";"):
            term = term.strip()
            if term in term_to_idx:
                go_labels[i, term_to_idx[term]] = 1.0

    return go_labels


def train_hierarchical_model(X_train, y1_train, y2_train,
                              X_val, y1_val, y2_val,
                              epochs=50, batch_size=64, lr=1e-3, device="cuda"):
    """
    Train the hierarchical EC classifier.

    Loss = Level1 CE loss + Level2 CE loss (per-class, weighted by main class prediction)
    """
    input_dim = X_train.shape[1]
    model = HierarchicalECClassifier(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss functions
    level1_criterion = nn.CrossEntropyLoss()
    # Level 2: one CrossEntropyLoss per main class
    level2_criteria = [
        nn.CrossEntropyLoss() for _ in range(6)
    ]

    # Data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y1_train), torch.LongTensor(y2_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(y1_val), torch.LongTensor(y2_val)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_val_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
    save_path = RESULTS_DIR / "best_hierarchical.pt"

    print(f"\nTraining Hierarchical EC Classifier for {epochs} epochs...")
    print(f"{'='*50}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y1_batch, y2_batch in train_loader:
            X_batch = X_batch.to(device)
            y1_batch = y1_batch.to(device)
            y2_batch = y2_batch.to(device)

            optimizer.zero_grad()
            level1_logits, level2_logits_list = model(X_batch)

            # Level 1 loss
            loss = level1_criterion(level1_logits, y1_batch)

            # Level 2 loss: apply each class-specific head only to
            # the samples belonging to that main class (via masks)
            for cls_idx in range(6):
                mask = (y1_batch == cls_idx)
                if mask.sum() == 0:
                    continue
                cls_logits = level2_logits_list[cls_idx][mask]      # (n, n_subclasses)
                cls_targets = y2_batch[mask]
                loss = loss + level2_criteria[cls_idx](cls_logits, cls_targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # Validation: hierarchical prediction
        model.eval()
        val_loss = 0.0
        all_main_preds, all_main_true = [], []
        with torch.no_grad():
            for X_batch, y1_batch, y2_batch in val_loader:
                X_batch = X_batch.to(device)
                y1_batch = y1_batch.to(device)

                level1_logits, _ = model(X_batch)
                val_loss += level1_criterion(level1_logits, y1_batch).item()
                preds = level1_logits.argmax(dim=1)
                all_main_preds.extend(preds.cpu().numpy())
                all_main_true.extend(y1_batch.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(all_main_true, all_main_preds, average="macro")
        val_acc = accuracy_score(all_main_true, all_main_preds)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)

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


def evaluate_hierarchical(X_test, y1_test, y2_test, device="cuda"):
    """Evaluate hierarchical model on test set."""
    input_dim = X_test.shape[1]
    model = HierarchicalECClassifier(input_dim=input_dim).to(device)
    checkpoint_path = RESULTS_DIR / "best_hierarchical.pt"

    if not checkpoint_path.exists():
        print(f"Error: {checkpoint_path} not found.")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        main_class, subclass_preds, ec_strings = model.predict_hierarchical(X_tensor)
        main_preds = main_class.cpu().numpy()

    acc = accuracy_score(y1_test, main_preds)
    f1 = f1_score(y1_test, main_preds, average="macro")
    report = classification_report(y1_test, main_preds, target_names=EC_CLASSES, output_dict=True)

    print(f"\n{'='*50}")
    print(f"TEST SET RESULTS — HIERARCHICAL EC CLASSIFIER")
    print(f"{'='*50}")
    print(f"Main Class Accuracy: {acc:.4f}")
    print(f"Main Class Macro F1: {f1:.4f}")
    print(classification_report(y1_test, main_preds, target_names=EC_CLASSES))

    # Show sample EC predictions
    print("Sample hierarchical predictions:")
    for i in range(min(10, len(ec_strings))):
        true_mc = y1_test[i] + 1
        print(f"  True: EC{true_mc}.-.-  |  Predicted: {ec_strings[i]}")

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "report": report,
        "predictions": main_preds,
        "ec_predictions": ec_strings,
    }


def train_multitask_model(X_train, y_ec_train, y_go_train,
                           X_val, y_ec_val, y_go_val,
                           epochs=50, batch_size=64, lr=1e-3,
                           go_weight=0.3, device="cuda"):
    """
    Train multi-task EC+GO model.

    Loss = EC loss + go_weight * GO loss (binary cross-entropy)
    """
    input_dim = X_train.shape[1]
    num_go_terms = y_go_train.shape[1]
    model = MultiTaskECGO(input_dim=input_dim, num_go_terms=num_go_terms).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ec_criterion = nn.CrossEntropyLoss()
    go_criterion = nn.BCEWithLogitsLoss()

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_ec_train), torch.FloatTensor(y_go_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), torch.LongTensor(y_ec_val), torch.FloatTensor(y_go_val)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_val_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}
    save_path = RESULTS_DIR / "best_multitask.pt"

    print(f"\nTraining Multi-task EC+GO Model for {epochs} epochs...")
    print(f"  GO auxiliary weight: {go_weight}")
    print(f"{'='*50}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_ec_batch, y_go_batch in train_loader:
            X_batch = X_batch.to(device)
            y_ec_batch = y_ec_batch.to(device)
            y_go_batch = y_go_batch.to(device)

            optimizer.zero_grad()
            ec_logits, go_logits = model(X_batch)

            loss = ec_criterion(ec_logits, y_ec_batch) + go_weight * go_criterion(go_logits, y_go_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_ec_batch, y_go_batch in val_loader:
                X_batch = X_batch.to(device)
                y_ec_batch = y_ec_batch.to(device)
                y_go_batch = y_go_batch.to(device)

                ec_logits, go_logits = model(X_batch)
                loss = ec_criterion(ec_logits, y_ec_batch) + go_weight * go_criterion(go_logits, y_go_batch)
                val_loss += loss.item()
                preds = ec_logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_ec_batch.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        val_acc = accuracy_score(all_labels, all_preds)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)

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


def evaluate_multitask(X_test, y_ec_test, y_go_test, device="cuda"):
    """Evaluate multi-task model on test set (EC task)."""
    input_dim = X_test.shape[1]
    num_go_terms = y_go_test.shape[1]
    model = MultiTaskECGO(input_dim=input_dim, num_go_terms=num_go_terms).to(device)
    checkpoint_path = RESULTS_DIR / "best_multitask.pt"

    if not checkpoint_path.exists():
        print(f"Error: {checkpoint_path} not found.")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        ec_logits, go_logits = model(X_tensor)
        preds = ec_logits.argmax(dim=1).cpu().numpy()
        go_preds = (torch.sigmoid(go_logits) > 0.5).cpu().numpy()

    acc = accuracy_score(y_ec_test, preds)
    f1 = f1_score(y_ec_test, preds, average="macro")
    report = classification_report(y_ec_test, preds, target_names=EC_CLASSES, output_dict=True)

    # GO metrics (micro F1 for multi-label)
    go_f1_micro = f1_score(y_go_test, go_preds, average="micro")

    print(f"\n{'='*50}")
    print(f"TEST SET RESULTS — MULTI-TASK EC+GO")
    print(f"{'='*50}")
    print(f"EC Accuracy: {acc:.4f}")
    print(f"EC Macro F1: {f1:.4f}")
    print(f"GO Micro F1: {go_f1_micro:.4f}")
    print(classification_report(y_ec_test, preds, target_names=EC_CLASSES))

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "go_f1_micro": go_f1_micro,
        "report": report,
        "predictions": preds,
    }


def save_all_results(sklearn_results, pytorch_results, history, hierarchical_result=None, multitask_result=None):
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

    if hierarchical_result is not None:
        output["pytorch_models"]["Hierarchical EC"] = {
            "accuracy": float(hierarchical_result["accuracy"]),
            "f1_macro": float(hierarchical_result["f1_macro"]),
        }

    if multitask_result is not None:
        output["pytorch_models"]["Multi-task EC+GO"] = {
            "accuracy": float(multitask_result["accuracy"]),
            "f1_macro": float(multitask_result["f1_macro"]),
            "go_f1_micro": float(multitask_result.get("go_f1_micro", 0)),
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

    # 3. PyTorch models (MLP + Residual)
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

    # 4. Hierarchical EC classifier
    print("\n" + "="*60)
    print("HIERARCHICAL EC CLASSIFIER (DEEPre-inspired)")
    print("="*60)
    csv_path = DATA_DIR / "enzymes_swissprot.csv"
    y1_train, y2_train = load_hierarchical_labels(csv_path)
    # Re-split using same indices (deterministic split by index)
    all_idx = np.arange(len(y1_train))
    trainval_idx, test_idx = train_test_split(
        all_idx, test_size=0.15, stratify=y1_train, random_state=42
    )
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=0.176, stratify=y1_train[trainval_idx], random_state=42
    )

    hierarchical_model, hier_hist = train_hierarchical_model(
        X_train, y1_train[train_idx], y2_train[train_idx],
        X_val, y1_train[val_idx], y2_train[val_idx],
        epochs=50, batch_size=64, lr=1e-3, device=device,
    )
    histories["hierarchical"] = hier_hist
    hierarchical_result = evaluate_hierarchical(
        X_test, y1_train[test_idx], y2_train[test_idx], device
    )

    # 5. Multi-task EC+GO model
    print("\n" + "="*60)
    print("MULTI-TASK EC + GO MODEL")
    print("="*60)
    go_labels = load_go_labels(csv_path)
    multitask_result = None

    # Only train the multi-task model if GO annotations are present
    if go_labels.sum() > 0:
        mt_model, mt_hist = train_multitask_model(
            X_train, y_train, go_labels[train_idx],
            X_val, y_val, go_labels[val_idx],
            epochs=50, batch_size=64, lr=1e-3, go_weight=0.3, device=device,
        )
        histories["multitask"] = mt_hist
        multitask_result = evaluate_multitask(
            X_test, y_test, go_labels[test_idx], device
        )
    else:
        print("Skipping multi-task training: GO annotations unavailable.")
        print("Re-run with 'python data/download_data.py' to fetch GO data.")

    # 6. Save everything
    save_all_results(sklearn_results, pytorch_results, histories, hierarchical_result, multitask_result)

    # 7. Print summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    all_results = {}
    for name, res in sklearn_results.items():
        all_results[name] = {"acc": res["accuracy"], "f1": res["f1_macro"]}
    for name, res in pytorch_results.items():
        if res:
            all_results[name.upper()] = {"acc": res["accuracy"], "f1": res["f1_macro"]}
    if hierarchical_result:
        all_results["HIERARCHICAL EC"] = {
            "acc": hierarchical_result["accuracy"], "f1": hierarchical_result["f1_macro"]
        }
    if multitask_result:
        all_results["MULTI-TASK EC+GO"] = {
            "acc": multitask_result["accuracy"], "f1": multitask_result["f1_macro"]
        }

    for name, metrics in sorted(all_results.items(), key=lambda x: x[1]["f1"], reverse=True):
        print(f"  {name:25s} | Acc: {metrics['acc']:.4f} | F1: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
