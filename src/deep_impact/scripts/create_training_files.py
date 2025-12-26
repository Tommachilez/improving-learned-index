import csv
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer

def load_doc_mapping(csv_path):
    """Loads doc_id -> document text mapping."""
    print(f"Loading documents from {csv_path}...")
    docs = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # key: doc_id, value: document
            if 'doc_id' in row and 'document' in row:
                docs[str(row['doc_id']).strip()] = row['document']
    return docs

def process_queries_mapping(input_csv, output_tsv):
    """Converts unique_query_mapping.csv to a headerless TSV."""
    print(f"Converting {input_csv} to {output_tsv}...")
    with open(input_csv, 'r', encoding='utf-8') as f_in, \
         open(output_tsv, 'w', encoding='utf-8', newline='') as f_out:

        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out, delimiter='\t')

        for row in tqdm(reader, desc="Processing Queries"):
            if 'query_id' in row and 'query' in row:
                writer.writerow([row['query_id'], row['query']])

def main():
    parser = argparse.ArgumentParser(description="Expand documents with unique query terms and truncate to 512 tokens.")

    # Inputs
    parser.add_argument("--doc_mapping", required=True, help="Path to unique_doc_mapping.csv")
    parser.add_argument("--query_mapping", required=True, help="Path to unique_query_mapping.csv")
    parser.add_argument("--pretokenized_queries", required=True, help="Path to queries_pretokenized.jsonl")

    # Outputs
    parser.add_argument("--output_queries_tsv", required=True, help="Output path for queries TSV")
    parser.add_argument("--output_docs_tsv", required=True, help="Output path for expanded documents TSV")

    # Config
    parser.add_argument("--model_name", default="xlm-roberta-base", help="Tokenizer model to use for length calculation.")
    parser.add_argument("--max_length", type=int, default=512, help="Max total tokens (doc + expansion)")

    args = parser.parse_args()

    # 1. Process Query Mapping (Simple Convert)
    process_queries_mapping(args.query_mapping, args.output_queries_tsv)

    # 2. Load Documents
    doc_map = load_doc_mapping(args.doc_mapping)

    # 3. Initialize Tokenizer
    print(f"Loading tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 4. Process Expansion
    print(f"Processing expansions from {args.pretokenized_queries}...")

    with open(args.pretokenized_queries, 'r', encoding='utf-8') as f_in, \
         open(args.output_docs_tsv, 'w', encoding='utf-8', newline='') as f_out:

        writer = csv.writer(f_out, delimiter='\t')

        for line in tqdm(f_in, desc="Expanding Documents"):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            pos_doc_id = str(entry.get('pos_doc_id', '')).strip()
            queries_list = entry.get('queries', [])

            # Retrieve original document
            original_doc = doc_map.get(pos_doc_id)

            if not original_doc:
                continue

            # Extract unique terms from query_seg
            unique_terms = set()
            for q in queries_list:
                seg = q.get('query_seg', '')
                if seg:
                    # Assuming query_seg is space-separated words/compounds
                    terms = seg.split()
                    unique_terms.update(terms)

            expansion_text = " ".join(unique_terms)

            # Tokenization & Truncation Logic
            # Goal: len(doc) + len(expansion) <= 512

            # 1. Tokenize expansion first (Priority: Keep generated terms)
            exp_tokens = tokenizer.tokenize(expansion_text)
            num_exp = len(exp_tokens)

            # 2. Calculate remaining budget for document
            budget_for_doc = args.max_length - num_exp

            if budget_for_doc <= 0:
                # Edge case: Expansion is massive (>512).
                # Strategy: Keep only expansion, truncated to max_length.
                final_tokens = exp_tokens[:args.max_length]
            else:
                # Tokenize document
                doc_tokens = tokenizer.tokenize(original_doc)

                # Truncate document to fit budget
                truncated_doc_tokens = doc_tokens[:budget_for_doc]

                # Combine: Doc + Expansion
                final_tokens = truncated_doc_tokens + exp_tokens

            # Decode back to string
            final_text = tokenizer.convert_tokens_to_string(final_tokens)

            # Write to TSV: doc_id, expanded_text
            writer.writerow([pos_doc_id, final_text])

    print(f"Done! Files saved to:\n- {args.output_queries_tsv}\n- {args.output_docs_tsv}")

if __name__ == "__main__":
    main()
