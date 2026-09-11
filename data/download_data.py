"""
Download enzyme sequences from UniProt SwissProt.
Task: 6-class classification (EC level 1)
  1 - Oxidoreductases
  2 - Transferases
  3 - Hydrolases
  4 - Lyases
  5 - Isomerases
  6 - Ligases

Filters: reviewed entries only, sequences <= 1022 residues (ESM-2 max length)
"""

import pandas as pd
import requests
import time
import os
import sys
from pathlib import Path


EC_CLASS_NAMES = {
    1: "Oxidoreductases",
    2: "Transferases",
    3: "Hydrolases",
    4: "Lyases",
    5: "Isomerases",
    6: "Ligases",
}

MAX_SEQ_LENGTH = 1022  # ESM-2 context limit
DATA_DIR = Path(__file__).parent


def query_uniprot(ec_class, max_results=2000):
    """
    Query UniProt REST API for reviewed enzymes of a given EC class.
    Returns list of dicts with sequence, accession, and metadata.
    """
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    query = f"(reviewed:true) AND (ec:{ec_class}.*.*) AND (length:[* TO {MAX_SEQ_LENGTH}])"

    params = {
        "query": query,
        "format": "json",
        "fields": "accession,sequence,ec,protein_name,organism_name,length",
        "size": min(max_results, 500),
    }

    proteins = []
    next_link = None

    while True:
        url = next_link or base_url
        response = requests.get(url, params=params if not next_link else None, timeout=30)

        if response.status_code != 200:
            print(f"  Warning: HTTP {response.status_code} for EC class {ec_class}")
            break

        data = response.json()
        results = data.get("results", [])

        for entry in results:
            seq = entry.get("sequence", {}).get("value", "")
            if seq and len(seq) <= MAX_SEQ_LENGTH and len(seq) >= 20:
                ec_list = entry.get("ec", [])
                proteins.append({
                    "accession": entry.get("primaryAccession", ""),
                    "sequence": seq,
                    "ec_class": ec_class,
                    "ec_number": ec_list[0] if ec_list else f"{ec_class}.-.-.-",
                    "ec_level1_name": EC_CLASS_NAMES[ec_class],
                    "protein_name": entry.get("proteinDescription", {})
                        .get("recommendedName", {})
                        .get("fullName", {})
                        .get("value", "Unknown") if entry.get("proteinDescription", {})
                        .get("recommendedName") else "Unknown",
                    "organism": entry.get("organism", {}).get("scientificName", "Unknown"),
                    "length": len(seq),
                })

        # Check for next page
        link_header = response.headers.get("Link", "")
        if 'rel="next"' in link_header:
            next_link = link_header.split(";")[0].strip("<>")
            params = None  # next link already has params
            time.sleep(0.5)
        else:
            break

        if len(proteins) >= max_results:
            break

    return proteins


def download_all_enzymes(target_per_class=1500):
    """Download enzymes for all 6 EC classes."""
    all_proteins = []

    for ec_class in range(1, 7):
        print(f"Downloading EC class {ec_class}: {EC_CLASS_NAMES[ec_class]}...")
        proteins = query_uniprot(ec_class, max_results=target_per_class)
        all_proteins.extend(proteins)
        print(f"  Collected {len(proteins)} sequences")
        time.sleep(1)  # respect rate limits

    df = pd.DataFrame(all_proteins)
    output_path = DATA_DIR / "enzymes_swissprot.csv"
    df.to_csv(output_path, index=False)

    print(f"\n{'='*50}")
    print(f"Total sequences downloaded: {len(df)}")
    print(f"\nClass distribution:")
    for ec_class in range(1, 7):
        count = len(df[df["ec_class"] == ec_class])
        print(f"  EC{ec_class} ({EC_CLASS_NAMES[ec_class]:16s}): {count:5d} sequences")

    print(f"\nLength statistics:")
    print(f"  Min: {df['length'].min()}")
    print(f"  Max: {df['length'].max()}")
    print(f"  Mean: {df['length'].mean():.0f}")
    print(f"  Median: {df['length'].median():.0f}")

    print(f"\nSaved to: {output_path}")
    return df


if __name__ == "__main__":
    df = download_all_enzymes(target_per_class=1500)
