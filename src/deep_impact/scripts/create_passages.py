import csv
import argparse
from tqdm import tqdm
import os

def sliding_window(text, window_size=350, stride=175):
    tokens = text.split()
    if not tokens: return []
    if len(tokens) <= window_size: return [text]
    
    windows = []
    for i in range(0, len(tokens), stride):
        chunk = " ".join(tokens[i : i + window_size])
        windows.append(chunk)
        if i + window_size >= len(tokens): break
    return windows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to expanded_collection.tsv")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    passages_path = os.path.join(args.output_dir, "passages.tsv")
    mapping_path = os.path.join(args.output_dir, "pid_mapping.txt")

    print(f"Processing {args.input}...")

    with open(args.input, 'r', encoding='utf-8') as f_in, \
         open(passages_path, 'w', encoding='utf-8', newline='') as f_pass, \
         open(mapping_path, 'w', encoding='utf-8') as f_map:

        reader = csv.reader(f_in, delimiter='\t')
        writer = csv.writer(f_pass, delimiter='\t')

        global_index = 0

        for row in tqdm(reader):
            if len(row) < 2: continue

            # Input format: [DocID, Text...]
            original_doc_id = row[0]
            # Handle cases where text might be split across columns or just col 1
            text = " ".join(row[1:]) 

            passages = sliding_window(text)

            for i, p in enumerate(passages):
                # 1. Write Passage (Format: 'IntID \t Text')
                # We use global_index here just to satisfy the TSV parser, though it's ignored.
                writer.writerow([global_index, p])

                # 2. Write Mapping (Line N -> RealID)
                # Format: "doc_123#0" (newline)
                real_passage_id = f"{original_doc_id}#{i}"
                f_map.write(f"{real_passage_id}\n")

                global_index += 1

    print(f"Done. Created {global_index} passages.")
    print(f"Files saved to:\n  - {passages_path}\n  - {mapping_path}")

if __name__ == "__main__":
    main()
