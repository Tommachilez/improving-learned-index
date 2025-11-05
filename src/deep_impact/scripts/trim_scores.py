#!/usr/bin/env python

"""
This script trims a "scores" file (gzipped pickle) to only include 
passage IDs (pid) that are present in a given "collection" file (tsv).

This is designed to fix data mismatch errors (like KeyError) by creating 
a new, aligned scores file that matches the collection.
"""

import gzip
import pickle
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

def load_valid_pids(collection_path: Path) -> set:
    """
    Loads all unique PIDs from the collection.tsv file into a set.
    Based on the logic in src/utils/datasets.py, the collection file
    is a TSV where the first column is the PID.
    """
    if not collection_path.exists():
        print(f"Error: Collection file not found at {collection_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading valid PIDs from collection: {collection_path}...")
    valid_pids = set()
    line_count = 0
    try:
        with open(collection_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading collection"):
                line_count += 1
                try:
                    # The PID is the first column
                    pid_str = line.split('\t')[0]
                    # The collection class stores PIDs as integers
                    valid_pids.add(int(pid_str))
                except (IndexError, ValueError):
                    print(f"Warning: Skipping malformed line {line_count}: {line.strip()}", file=sys.stderr)

    except Exception as e:
        print(f"An error occurred while reading {collection_path}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(valid_pids)} unique valid PIDs in {line_count} lines.")
    return valid_pids

def load_scores_data(scores_path: Path) -> dict:
    """
    Loads the gzipped pickle file containing the scores.
    Based on src/utils/datasets.py.
    """
    if not scores_path.exists():
        print(f"Error: Scores file not found at {scores_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading scores data from {scores_path}...")
    try:
        with gzip.open(scores_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Scores data loaded. Found scores for {len(data)} QIDs.")
        return data
    except Exception as e:
        print(f"An error occurred while loading {scores_path}: {e}", file=sys.stderr)
        sys.exit(1)

def trim_scores(scores_data: dict, valid_pids: set) -> dict:
    """
    Creates a new scores dictionary, filtering out PIDs not in the valid_pids set.
    """
    print("Trimming scores data...")
    trimmed_scores_data = {}
    original_pid_entries = 0
    trimmed_pid_entries = 0

    for qid, pid_score_map in tqdm(scores_data.items(), desc="Trimming QIDs"):
        original_pid_entries += len(pid_score_map)

        # Filter the inner dictionary to keep only valid PIDs
        new_pid_score_map = {
            pid: score for pid, score in pid_score_map.items() if pid in valid_pids
        }

        # Only add the QID back if it still has valid passages
        if new_pid_score_map:
            trimmed_scores_data[qid] = new_pid_score_map
            trimmed_pid_entries += len(new_pid_score_map)

    removed_entries = original_pid_entries - trimmed_pid_entries
    print("\n--- Trimming Stats ---")
    print(f"Original PID-Score entries: {original_pid_entries:>12,}")
    print(f" Trimmed PID-Score entries: {trimmed_pid_entries:>12,}")
    print(f"   Removed entries (missing): {removed_entries:>12,}")

    return trimmed_scores_data

def save_trimmed_scores(data: dict, output_path: Path):
    """
    Saves the new trimmed dictionary to a gzipped pickle file.
    """
    print(f"\nSaving trimmed data to {output_path}...")
    try:
        with gzip.open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print("Save complete.")
    except Exception as e:
        print(f"An error occurred while saving {output_path}: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Trim a scores.pkl.gz file based on PIDs from a collection.tsv file.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--collection_path",
        type=Path,
        required=True,
        help="Path to the master collection.tsv file (e.g., 'collection.tsv')."
    )

    parser.add_argument(
        "--scores_path",
        type=Path,
        required=True,
        help="Path to the input scores file to be trimmed (e.g., 'scores.pkl.gz')."
    )

    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to save the new, trimmed scores file (e.g., 'scores.trimmed.pkl.gz')."
    )

    args = parser.parse_args()

    # Step 1: Get the set of all valid PIDs from the collection
    valid_pids = load_valid_pids(args.collection_path)

    # Step 2: Load the scores file
    scores_data = load_scores_data(args.scores_path)

    # Step 3: Trim the scores data
    trimmed_data = trim_scores(scores_data, valid_pids)

    # Step 4: Save the new trimmed data
    save_trimmed_scores(trimmed_data, args.output_path)

    print("\nDone. Your new scores file is ready for training.")

if __name__ == "__main__":
    main()
