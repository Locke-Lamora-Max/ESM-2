"""
Multiple classifier architectures for enzyme EC number prediction.

Compares:
  1. Logistic Regression (linear baseline)
  2. Random Forest (ensemble baseline)
  3. Gradient Boosting (strong baseline)
  4. MLP Classifier (neural network on frozen embeddings)
  5. Residual MLP (deeper architecture)
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score

EC_CLASSES = [f"EC{i}" for i in range(1, 7)]


# ============================================================
# PyTorch Models
# ============================================================

class MLPClassifier(nn.Module):
    """
    Multi-layer perceptron for classification on ESM-2 embeddings.

    Architecture:
      Input (480) → Linear → BN → ReLU → Dropout → Linear → BN → ReLU → Dropout → Output (6)
    """
    def __init__(self, input_dim=480, hidden_dims=(256, 128), num_classes=6, dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ResidualMLP(nn.Module):
    """
    Residual MLP — shows deeper architecture understanding.

    Uses residual connections to allow gradient flow through deeper networks.
    """
    def __init__(self, input_dim=480, hidden_dim=256, num_classes=6, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.block1 = self._make_block(hidden_dim, dropout)
        self.block2 = self._make_block(hidden_dim, dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _make_block(self, dim, dropout):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = torch.relu(x + self.block1(x))
        x = torch.relu(x + self.block2(x))
        return self.classifier(x)


# ============================================================
# Scikit-learn Baselines
# ============================================================

def train_sklearn_baselines(X_train, y_train, X_test, y_test):
    """
    Train and evaluate classical ML baselines on ESM-2 embeddings.

    Returns dict of {model_name: {accuracy, f1_macro, report_dict}}
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", multi_class="multinomial"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\n{'='*40}")
        print(f"Training: {name}")
        print(f"{'='*40}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        report = classification_report(
            y_test, y_pred, target_names=EC_CLASSES, output_dict=True
        )

        results[name] = {
            "accuracy": acc,
            "f1_macro": f1,
            "report": report,
        }

        print(f"Accuracy: {acc:.4f}")
        print(f"Macro F1: {f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=EC_CLASSES))

    return results
