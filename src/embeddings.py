"""
Extract protein embeddings using ESM-2 (Meta's protein language model).

ESM-2 architecture:
  - Transformer encoder trained on 250M+ protein sequences (UniRef90)
  - Learns evolutionary and functional patterns from amino acid sequences
  - Outputs per-residue embeddings that can be pooled to protein-level

Pooling strategies:
  - Mean pooling (default): average over residue embeddings
  - Attention pooling: learned attention weights over residues, allowing
    the model to focus on functionally important positions (e.g., active sites)

This module uses esm2_t12_35M_UR50D (35M params, 12 layers, 480-dim)
as the primary model — good balance of quality and speed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
ESM2_MODEL = "esm2_t12_35M_UR50D"  # 35M params, 480-dim embeddings


class AttentionPool(nn.Module):
    """
    Learned attention-weighted pooling over residue embeddings.

    Instead of simple mean-pooling (which treats all residues equally),
    this module learns a small attention network that assigns importance
    weights to each residue. This allows the model to focus on
    functionally critical positions such as active sites and binding residues.

    Architecture:
      Residue embeddings (L, D) → Linear(D, D//2) → Tanh → Linear(D//2, 1)
      → Softmax weights (L, 1) → Weighted sum → Protein embedding (D,)
    """
    def __init__(self, embed_dim=480):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, residue_embeds):
        """
        Args:
            residue_embeds: (batch, seq_len, embed_dim) per-residue embeddings
        Returns:
            protein_embed: (batch, embed_dim) attention-pooled embedding
            attention_weights: (batch, seq_len) normalized attention weights
        """
        # Compute attention scores: (batch, seq_len, 1)
        scores = self.attention(residue_embeds)
        # Normalize: (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)
        # Weighted sum: (batch, embed_dim)
        protein_embed = (weights * residue_embeds).sum(dim=1)
        return protein_embed, weights.squeeze(-1)


def load_model(device="cuda"):
    """Load ESM-2 model and alphabet."""
    print(f"Loading {ESM2_MODEL}...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(ESM2_MODEL)
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    embedding_dim = getattr(model, "embed_dim", None) or getattr(getattr(model, "args", None), "embed_dim", 480)
    num_layers = getattr(model, "num_layers", None) or getattr(getattr(model, "args", None), "num_layers", 12)

    print(f"  Model loaded: {embedding_dim}-dim embeddings, {num_layers} layers")
    return model, alphabet, batch_converter, embedding_dim, num_layers


def extract_embeddings_batch(sequences, batch_size=32, device="cuda", pooling="mean"):
    """
    Extract protein-level embeddings from ESM-2.

    For each protein:
      1. Tokenize amino acid sequence
      2. Forward pass through ESM-2 transformer
      3. Take last hidden layer representations
      4. Pool over sequence length:
         - "mean": average over residues (exclude BOS/EOS)
         - "attention": learned attention-weighted sum (requires trained AttentionPool)
      5. Result: single vector of dimension 480

    Args:
        sequences: list of amino acid strings (e.g., ["MKFLILLFN...", ...])
        batch_size: number of sequences per forward pass
        device: 'cuda' or 'cpu'
        pooling: "mean" or "attention"

    Returns:
        numpy array of shape (n_sequences, embedding_dim)
    """
    model, alphabet, batch_converter, embedding_dim, num_layers = load_model(device)

    # Load attention pool if using attention pooling
    attention_pool = None
    if pooling == "attention":
        attention_pool = AttentionPool(embedding_dim).to(device)
        attn_path = Path(__file__).parent.parent / "results" / "attention_pool.pt"
        if attn_path.exists():
            attention_pool.load_state_dict(torch.load(attn_path, map_location=device, weights_only=True))
            print(f"  Loaded trained attention pool from {attn_path}")
        else:
            print(f"  Warning: No trained attention pool found at {attn_path}")
            print(f"  Using untrained attention weights (random pooling)")
        attention_pool.eval()

    all_embeddings = []
    total = len(sequences)

    for i in tqdm(range(0, total, batch_size), desc="Extracting embeddings"):
        batch_seqs = sequences[i:i + batch_size]
        batch_labels = [f"seq_{j}" for j in range(len(batch_seqs))]

        # Prepare input for ESM-2
        data = [(batch_labels[j], batch_seqs[j]) for j in range(len(batch_seqs))]
        batch_labels_out, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        # Forward pass
        with torch.no_grad():
            results = model(
                batch_tokens,
                repr_layers=[num_layers],
                return_contacts=False,
            )

        # Extract last layer representations
        token_reps = results["representations"][num_layers]

        if pooling == "attention":
            # Collect per-sequence residue embeddings, pad, and pool
            seq_embeds_list = []
            for j in range(len(batch_seqs)):
                seq_len = len(batch_seqs[j])
                emb = token_reps[j, 1:seq_len + 1]  # (seq_len, 480)
                seq_embeds_list.append(emb)

            # Pad to max length in batch
            max_len = max(e.shape[0] for e in seq_embeds_list)
            padded = torch.zeros(len(batch_seqs), max_len, embedding_dim, device=device)
            mask = torch.zeros(len(batch_seqs), max_len, dtype=torch.bool, device=device)
            for j, emb in enumerate(seq_embeds_list):
                padded[j, :emb.shape[0]] = emb
                mask[j, :emb.shape[0]] = True

            # Apply attention pooling with masking
            with torch.no_grad():
                scores = attention_pool.attention(padded)  # (batch, max_len, 1)
                scores = scores.squeeze(-1)  # (batch, max_len)
                scores[~mask] = float('-inf')
                weights = F.softmax(scores, dim=1)  # (batch, max_len)
                weights[~mask] = 0.0
                protein_embs = (weights.unsqueeze(-1) * padded).sum(dim=1)  # (batch, 480)

            for j in range(len(batch_seqs)):
                all_embeddings.append(protein_embs[j].cpu().numpy())
        else:
            # Mean pool for each sequence
            for j in range(len(batch_seqs)):
                seq_len = len(batch_seqs[j])
                emb = token_reps[j, 1:seq_len + 1].mean(dim=0)
                all_embeddings.append(emb.cpu().numpy())

    return np.array(all_embeddings, dtype=np.float32)


def extract_and_save(pooling="mean"):
    """Full pipeline: load CSV → extract embeddings → save to .npy"""
    csv_path = DATA_DIR / "enzymes_swissprot.csv"
    suffix = "_attention" if pooling == "attention" else ""
    output_path = DATA_DIR / f"esm2_embeddings{suffix}.npy"

    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run download_data.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} sequences from {csv_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    embeddings = extract_embeddings_batch(
        df["sequence"].tolist(),
        batch_size=32,
        device=device,
        pooling=pooling,
    )

    np.save(output_path, embeddings)
    print(f"\nSaved embeddings: shape={embeddings.shape}, path={output_path}")

    # Also save labels separately for convenience
    labels_path = DATA_DIR / "labels.npy"
    labels = df["ec_class"].values - 1  # 0-indexed for PyTorch
    np.save(labels_path, labels)
    print(f"Saved labels: shape={labels.shape}")

    return embeddings


if __name__ == "__main__":
    extract_and_save()
