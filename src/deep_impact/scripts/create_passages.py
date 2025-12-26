import csv
import json
import argparse
from tqdm import tqdm
import os

def sliding_window(text, window_size=350, stride=175):
    tokens = text.split()
    if not tokens:
        return []

    if len(tokens) <= window_size:
        return [text]

    windows = []
    for i in range(0, len(tokens), stride):
        chunk = " ".join(tokens[i : i + window_size])
        windows.append(chunk)
        if i + window_size >= len(tokens):
            break
    return windows

def load_expansion_terms(queries_path):
    """
    Loads pre-tokenized queries from JSONL and extracts unique terms per document.
    Expected JSONL format:
    {
        "pos_doc_id": "...", 
        "queries": [
            {"query_raw": "...", "query_seg": "term1 term2 ..."},
            ...
        ]
    }
    """
    print(f"Loading expansion terms from {queries_path}...")
    doc_expansions = {}

    try:
        with open(queries_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Parsing queries"):
                if not line.strip(): continue

                try:
                    data = json.loads(line)
                    # Ensure doc_id is string for consistent matching
                    doc_id = str(data.get('pos_doc_id', '')).strip()
                    queries = data.get('queries', [])

                    if not doc_id: continue

                    unique_terms = set()
                    for q in queries:
                        # Use query_seg for the actual terms
                        seg = q.get('query_seg', '')
                        if seg:
                            # Split by whitespace to get tokens
                            unique_terms.update(seg.split())

                    if unique_terms:
                        doc_expansions[doc_id] = " ".join(sorted(unique_terms))

                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Error: Queries file {queries_path} not found.")
        exit(1)

    print(f"Loaded expansion terms for {len(doc_expansions)} documents.")
    return doc_expansions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Path to doc_mapping.csv (must have 'doc_id', 'document' headers)")
    parser.add_argument("--queries_jsonl", type=str, required=True, help="Path to queries_pretokenized.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files")
    parser.add_argument("--window", type=int, default=300, help="Sliding window size (words)")
    parser.add_argument("--stride", type=int, default=150, help="Sliding window stride (words)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    passages_path = os.path.join(args.output_dir, "passages.tsv")
    mapping_path = os.path.join(args.output_dir, "pid_mapping.txt")

    # 1. Load Expansions
    doc_expansions = load_expansion_terms(args.queries_jsonl)

    # 2. Process Documents
    print(f"Processing documents from {args.input_csv}...")

    with open(args.input_csv, 'r', encoding='utf-8') as f_in, \
         open(passages_path, 'w', encoding='utf-8', newline='') as f_pass, \
         open(mapping_path, 'w', encoding='utf-8') as f_map:

        reader = csv.DictReader(f_in)
        if 'doc_id' not in reader.fieldnames or 'document' not in reader.fieldnames:
            print(f"Error: CSV must contain 'doc_id' and 'document' columns. Found: {reader.fieldnames}")
            exit(1)

        writer = csv.writer(f_pass, delimiter='\t')

        global_index = 0

        for row in tqdm(reader, desc="Creating passages"):
            doc_id = str(row['doc_id']).strip()
            text = row['document']

            if not text: continue

            # Get expansion terms for this doc (if any)
            expansion_text = doc_expansions.get(doc_id, "")

            # Create Windows
            passages = sliding_window(text, args.window, args.stride)

            for i, p in enumerate(passages):
                # Append expansion terms to EVERY passage
                # Structure: [Passage Text] [Expansion Terms]
                if expansion_text:
                    expanded_passage = f"{p} {expansion_text}"
                else:
                    expanded_passage = p

                # 1. Write to Passages TSV (IntID \t Text) for DeeperImpact
                writer.writerow([global_index, expanded_passage])

                # 2. Write to Mapping File (doc_id#passage_idx) for Evaluation
                f_map.write(f"{doc_id}#{i}\n")

                global_index += 1

    print(f"Done. Processed {global_index} total passages.")
    print(f"Outputs saved to:\n  - Passages: {passages_path}\n  - Mapping:  {mapping_path}")

if __name__ == "__main__":
    main()
