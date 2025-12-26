import csv
import json
import argparse
import os
from collections import Counter
from tqdm import tqdm


def sliding_window(text, window_size=250, stride=100):
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


def load_expansion_terms(queries_path, max_terms=100):
    """
    Loads pre-tokenized queries, counts term frequency, and selects top-K terms.
    """
    print(f"Loading expansion terms from {queries_path} with max_terms={max_terms}...")
    doc_expansions = {}

    try:
        with open(queries_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Parsing queries"):
                if not line.strip(): continue

                try:
                    data = json.loads(line)
                    doc_id = str(data.get('pos_doc_id', '')).strip()
                    queries = data.get('queries', [])

                    if not doc_id: continue

                    # Use Counter to track term importance
                    term_counts = Counter()
                    for q in queries:
                        seg = q.get('query_seg', '')
                        if seg:
                            term_counts.update(seg.split())

                    if term_counts:
                        # Select top K most frequent terms
                        top_terms = [term for term, count in term_counts.most_common(max_terms)]
                        # Keeping frequency order places most important terms first (better for truncation)
                        doc_expansions[doc_id] = " ".join(top_terms)

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
    parser.add_argument("--window", type=int, default=250, help="Sliding window size (words)")
    parser.add_argument("--stride", type=int, default=100, help="Sliding window stride (words)")
    parser.add_argument("--max_expansion_len", type=int, default=100, help="Max expansion terms to append")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    passages_path = os.path.join(args.output_dir, "passages.tsv")
    mapping_path = os.path.join(args.output_dir, "pid_mapping.txt")

    # 1. Load Expansions
    doc_expansions = load_expansion_terms(args.queries_jsonl, max_terms=args.max_expansion_len)

    # 2. Process Documents
    print(f"Processing documents from {args.input_csv}...")

    with open(args.input_csv, 'r', encoding='utf-8') as f_in, \
         open(passages_path, 'w', encoding='utf-8', newline='') as f_pass, \
         open(mapping_path, 'w', encoding='utf-8') as f_map:

        reader = csv.DictReader(f_in)
        if 'doc_id' not in reader.fieldnames or 'document' not in reader.fieldnames:
            print(f"Error: CSV must contain 'doc_id' and 'document' columns. Found: {reader.fieldnames}")
            exit(1)

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

                # SANITIZE: Replace tabs and newlines to prevent format corruption
                clean_passage = expanded_passage.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')

                # Write manually: ID \t Text \n
                f_pass.write(f"{global_index}\t{clean_passage}\n")

                f_map.write(f"{doc_id}#{i}\n")

                global_index += 1

    print(f"Done. Processed {global_index} total passages.")
    print(f"Outputs saved to:\n  - Passages: {passages_path}\n  - Mapping:  {mapping_path}")

if __name__ == "__main__":
    main()
