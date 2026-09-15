"""
Multiple classifier architectures for enzyme EC number prediction.

Compares:
  1. Logistic Regression (linear baseline)
  2. Random Forest (ensemble baseline)
  3. Gradient Boosting (strong baseline)
  4. MLP Classifier (neural network on frozen embeddings)
  5. Residual MLP (deeper architecture)
  6. Hierarchical EC Classifier (level-by-level, inspired by DEEPre)
  7. Multi-task EC+GO (shared backbone with GO auxiliary head)
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score

EC_CLASSES = [f"EC{i}" for i in range(1, 7)]

# EC hierarchy: level 2 subclasses per main class
# (number of subclasses varies per main class)
EC_SUBCLASS_COUNTS = {
    1: 28,   # EC1 (Oxidoreductases) subclasses 1-28
    2: 26,   # EC2 (Transferases) subclasses 1-26
    3: 13,   # EC3 (Hydrolases) subclasses 1-13
    4: 8,    # EC4 (Lyases) subclasses 1-8
    5: 6,    # EC5 (Isomerases) subclasses 1-6
    6: 7,    # EC6 (Ligases) subclasses 1-7
}

# Top GO terms for auxiliary task (most frequent across SwissProt)
TOP_GO_TERMS = [
    "GO:0005524",   # ATP binding
    "GO:0003674",   # molecular_function (root)
    "GO:0005488",   # binding
    "GO:0003824",   # catalytic activity
    "GO:0006468",   # protein phosphorylation
    "GO:0005737",   # cytoplasm
    "GO:0016020",   # membrane
    "GO:0005886",   # plasma membrane
    "GO:0005515",   # protein binding
    "GO:0008270",   # zinc ion binding
    "GO:0046872",   # metal ion binding
    "GO:0005634",   # nucleus
    "GO:0005829",   # cytosol
    "GO:0016021",   # integral component of membrane
    "GO:0005783",   # endoplasmic reticulum
    "GO:0009986",   # cell surface
    "GO:0045087",   # innate immune response
    "GO:0006915",   # apoptotic process
    "GO:0042981",   # regulation of apoptotic process
    "GO:0007165",   # signal transduction
]


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


class HierarchicalECClassifier(nn.Module):
    """
    Hierarchical EC number prediction — level-by-level classification.

    Inspired by DEEPre (Li et al., 2018), this model predicts EC numbers
    in a tree-structured manner:
      - Level 1: Main class (EC 1-6) — 6 classes
      - Level 2: Subclass conditioned on main class — variable per class

    At inference time, the model first predicts the main class, then uses
    a class-specific head to predict the subclass. This mirrors the EC
    classification hierarchy and allows the model to learn fine-grained
    distinctions within each main class.

    Architecture:
      Shared backbone: 480 → Linear(256) → BN → ReLU → Dropout → Linear(128)
      Level 1 head: 128 → Linear(6)
      Level 2 heads: 128 → Linear(n_subclasses) — one per main class
    """
    def __init__(self, input_dim=480, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Level 1: main class (6 classes)
        self.level1_head = nn.Linear(hidden_dim // 2, 6)

        # Level 2: subclass heads (one per main class)
        self.level2_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, count)
            for count in EC_SUBCLASS_COUNTS.values()
        ])

    def forward(self, x):
        """Forward pass for training — returns both level predictions."""
        features = self.backbone(x)
        level1_logits = self.level1_head(features)
        level2_logits = [head(features) for head in self.level2_heads]
        return level1_logits, level2_logits

    def predict_hierarchical(self, x):
        """
        Hierarchical inference: predict main class, then subclass.

        Returns:
            main_class: (batch,) predicted main class indices (0-5)
            subclass: (batch,) predicted subclass indices
            ec_string: list of EC number strings like "2.7.1.1"
        """
        features = self.backbone(x)
        # Level 1: predict main class
        level1_logits = self.level1_head(features)
        main_class = level1_logits.argmax(dim=1)

        # Level 2: predict subclass conditioned on main class
        subclass_preds = []
        for i in range(x.shape[0]):
            cls_idx = main_class[i].item()
            level2_logits = self.level2_heads[cls_idx](features[i:i+1])
            subclass_preds.append(level2_logits.argmax(dim=1).item() + 1)  # 1-indexed

        subclass_preds = torch.tensor(subclass_preds, device=x.device)

        # Build EC strings (main_class.subclass.-.-)
        ec_strings = []
        for i in range(x.shape[0]):
            mc = main_class[i].item() + 1  # 1-indexed EC class
            sc = subclass_preds[i].item()
            ec_strings.append(f"{mc}.{sc}.-.-")

        return main_class, subclass_preds, ec_strings


class MultiTaskECGO(nn.Module):
    """
    Multi-task model: shared ESM-2 embedding backbone with dual heads.

    Jointly predicts:
      - Primary task: EC main class (6 classes)
      - Auxiliary task: GO term presence (multi-label, 20 terms)

    The GO auxiliary task provides regularization and forces the shared
    representation to capture broader functional signal beyond just EC
    classification. This is motivated by Xin Gao's group's work on
    DeepGO and protein function prediction.

    Architecture:
      Shared backbone: 480 → Linear(256) → BN → ReLU → Dropout → Linear(128)
      EC head: 128 → Linear(6)
      GO head: 128 → Linear(20) with sigmoid
    """
    def __init__(self, input_dim=480, hidden_dim=256, num_go_terms=20, dropout=0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.ec_head = nn.Linear(hidden_dim // 2, 6)
        self.go_head = nn.Linear(hidden_dim // 2, num_go_terms)

    def forward(self, x):
        """Returns EC logits and GO logits."""
        features = self.backbone(x)
        ec_logits = self.ec_head(features)
        go_logits = self.go_head(features)
        return ec_logits, go_logits


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
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=300, max_depth=5, learning_rate=0.1,
            random_state=42
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
