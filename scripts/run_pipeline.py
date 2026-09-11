"""
ESM-2 Enzyme Function Prediction — Main Runner

Usage:
    python scripts/run_pipeline.py              # Run full pipeline
    python scripts/run_pipeline.py --download   # Download data only
    python scripts/run_pipeline.py --embeddings # Extract embeddings only
    python scripts/run_pipeline.py --train      # Train models only
    python scripts/run_pipeline.py --evaluate   # Evaluate + visualize only
"""

import argparse
import sys
import os
import time
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_download():
    print("\n" + "="*60)
    print("STEP 1: Downloading enzyme data from UniProt")
    print("="*60)
    from data.download_data import download_all_enzymes
    download_all_enzymes(target_per_class=1500)


def run_embeddings():
    print("\n" + "="*60)
    print("STEP 2: Extracting ESM-2 embeddings")
    print("="*60)
    from src.embeddings import extract_and_save
    extract_and_save()


def run_training():
    print("\n" + "="*60)
    print("STEP 3: Training classifiers")
    print("="*60)
    from src.train import main
    main()


def run_evaluate():
    print("\n" + "="*60)
    print("STEP 4: Generating figures")
    print("="*60)
    from src.visualize import generate_all_figures
    generate_all_figures()


def main():
    parser = argparse.ArgumentParser(description="ESM-2 Enzyme Prediction Pipeline")
    parser.add_argument("--download", action="store_true", help="Download data only")
    parser.add_argument("--embeddings", action="store_true", help="Extract embeddings only")
    parser.add_argument("--train", action="store_true", help="Train models only")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate and visualize only")
    args = parser.parse_args()

    any_flag = args.download or args.embeddings or args.train or args.evaluate

    start_time = time.time()

    if not any_flag or args.download:
        run_download()

    if not any_flag or args.embeddings:
        run_embeddings()

    if not any_flag or args.train:
        run_training()

    if not any_flag or args.evaluate:
        run_evaluate()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Pipeline complete! Total time: {elapsed/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
