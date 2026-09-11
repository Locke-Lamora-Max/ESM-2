"""
Extract protein embeddings using ESM-2 (Meta's protein language model).

ESM-2 architecture:
  - Transformer encoder trained on 250M+ protein sequences (UniRef90)
  - Learns evolutionary and functional patterns from amino acid sequences
  - Outputs per-residue embeddings that can be pooled to protein-level

This module uses esm2_t12_35M_UR50D (35M params, 12 layers, 480-dim)
as the primary model — good balance of quality and speed.
"""

import torch
import esm
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
ESM2_MODEL = "esm2_t12_35M_UR50D"  # 35M params, 480-dim embeddings


def load_model(device="cuda"):
    """Load ESM-2 model and alphabet."""
    print(f"Loading {ESM2_MODEL}...")
    model, alphabet = esm.pretrained.load_model_and_alphabet(ESM2_MODEL)
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    embedding_dim = model.args.embed_dim
    num_layers = model.args.num_layers

    print(f"  Model loaded: {embedding_dim}-dim embeddings, {num_layers} layers")
    return model, alphabet, batch_converter, embedding_dim, num_layers


def extract_embeddings_batch(sequences, batch_size=32, device="cuda"):
    """
    Extract mean-pooled embeddings from ESM-2.

    For each protein:
      1. Tokenize amino acid sequence
      2. Forward pass through ESM-2 transformer
      3. Take last hidden layer representations
      4. Mean pool over sequence length (exclude BOS/EOS tokens)
      5. Result: single vector of dimension 480

    Args:
        sequences: list of amino acid strings (e.g., ["MKFLILLFN...", ...])
        batch_size: number of sequences per forward pass
        device: 'cuda' or 'cpu'

    Returns:
        numpy array of shape (n_sequences, embedding_dim)
    """
    model, alphabet, batch_converter, embedding_dim, num_layers = load_model(device)

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

        # Mean pool for each sequence
        for j in range(len(batch_seqs)):
            seq_len = len(batch_seqs[j])
            # tokens[0] = BOS, tokens[1:seq_len+1] = sequence, tokens[seq_len+1] = EOS
            emb = token_reps[j, 1:seq_len + 1].mean(dim=0)
            all_embeddings.append(emb.cpu().numpy())

    return np.array(all_embeddings, dtype=np.float32)


def extract_and_save():
    """Full pipeline: load CSV → extract embeddings → save to .npy"""
    csv_path = DATA_DIR / "enzymes_swissprot.csv"
    output_path = DATA_DIR / "esm2_embeddings.npy"

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
