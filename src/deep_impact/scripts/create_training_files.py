import csv
import json
import argparse
from collections import defaultdict, Counter
from tqdm import tqdm
from transformers import AutoTokenizer

def sanitize_text(text):
    if not text:
        return ""
    # split() without arguments splits by any whitespace (including \n, \t, \r)
    return " ".join(text.split()).strip()

def load_raw_docs(csv_path):
    """Loads doc_id -> raw document text from CSV."""
    print(f"Loading raw documents from CSV: {csv_path}...")
    docs = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'doc_id' in row and 'document' in row:
                docs[str(row['doc_id']).strip()] = row['document']
    return docs

def load_pretokenized_docs(jsonl_path):
    """Loads doc_id -> pretokenized content (space-separated tokens) for deduplication."""
    print(f"Loading pretokenized docs for filtering from: {jsonl_path}...")
    docs = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line)
                doc_id = str(entry.get('id', '')).strip()
                content = entry.get('contents', '')
                if doc_id:
                    docs[doc_id] = content
            except json.JSONDecodeError:
                continue
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
                clean_q = sanitize_text(row['query'])
                writer.writerow([row['query_id'], clean_q])

def main():
    parser = argparse.ArgumentParser(description="Expand raw documents with unique query terms (deduplicated & cleaned).")

    # Inputs
    parser.add_argument("--doc_mapping", required=True, help="Path to unique_doc_mapping.csv (Raw Documents)")
    parser.add_argument("--pretokenized_doc", required=True, help="Path to pretokenized document JSONL (For filtering)")
    parser.add_argument("--query_mapping", required=True, help="Path to unique_query_mapping.csv")
    parser.add_argument("--pretokenized_queries", required=True, help="Path to queries_pretokenized.jsonl")

    # Outputs
    parser.add_argument("--output_queries_tsv", required=True, help="Output path for queries TSV")
    parser.add_argument("--output_docs_tsv", required=True, help="Output path for expanded documents TSV (Final Model Input)")
    parser.add_argument("--output_expansion_csv", required=True, help="Output path for Expansion Terms CSV (doc_id, added_terms)")

    # Config
    parser.add_argument("--model_name", default="xlm-roberta-base", help="Tokenizer model to use for length calculation.")
    parser.add_argument("--max_length", type=int, default=512, help="Max total tokens (doc + expansion)")
    parser.add_argument("--max_expansion_terms", type=int, default=100, help="Max unique expansion terms to append (Top-K frequency).")

    args = parser.parse_args()

    # 1. Process Query Mapping
    process_queries_mapping(args.query_mapping, args.output_queries_tsv)

    # 2. Load Documents (Both Raw and Pretokenized)
    raw_docs_map = load_raw_docs(args.doc_mapping)
    pretok_docs_map = load_pretokenized_docs(args.pretokenized_doc)

    # 3. Aggregate Query Terms by Document ID
    print(f"Aggregating query terms from {args.pretokenized_queries}...")

    # CHANGE: Use Counter to track frequency, not just a Set
    doc_expansions = defaultdict(Counter)

    with open(args.pretokenized_queries, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading Queries JSONL"):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            pos_doc_id = str(entry.get('pos_doc_id', '')).strip()
            if not pos_doc_id:
                continue

            queries_list = entry.get('queries', [])

            # Concatenate terms from all queries for this entry
            for q in queries_list:
                seg = q.get('query_seg', '')
                if seg:
                    terms = seg.split()
                    doc_expansions[pos_doc_id].update(terms)

    # 4. Initialize Tokenizer
    print(f"Loading tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 5. Build Expanded Documents
    print(f"Building expanded documents...")
    print(f"- Expanded Docs TSV: {args.output_docs_tsv}")
    print(f"- Expansion Terms CSV: {args.output_expansion_csv}")
    print(f"- Max Expansion Terms: {args.max_expansion_terms}")

    with open(args.output_docs_tsv, 'w', encoding='utf-8', newline='') as f_doc_out, \
         open(args.output_expansion_csv, 'w', encoding='utf-8', newline='') as f_exp_out:

        doc_writer = csv.writer(f_doc_out, delimiter='\t')
        exp_writer = csv.writer(f_exp_out)

        # Header for the auxiliary CSV
        exp_writer.writerow(["doc_id", "expansion_terms"])

        processed_count = 0

        # Iterate over documents that have expansions
        for doc_id, term_counts in tqdm(doc_expansions.items(), desc="Expanding"):

            # We need the RAW text to be the base
            raw_doc_text = raw_docs_map.get(doc_id)

            # We need the PRETOKENIZED text to check for duplicates
            pretok_doc_text = pretok_docs_map.get(doc_id)

            if not raw_doc_text:
                continue # Cannot expand if we don't have the original doc

            # --- DEDUPLICATION LOGIC ---
            existing_terms = set()
            if pretok_doc_text:
                existing_terms = set(pretok_doc_text.split())
            else:
                # Fallback to raw split if pretokenized is missing
                existing_terms = set(raw_doc_text.split())

            # --- SELECTION LOGIC (Top-K Filtered) ---
            # 1. Sort by frequency (Counter.most_common returns sorted list)
            # 2. Filter: Only keep terms NOT in document
            # 3. Limit: Stop after max_expansion_terms

            selected_terms = []
            for term, count in term_counts.most_common():
                if term not in existing_terms:
                    selected_terms.append(term)

                if len(selected_terms) >= args.max_expansion_terms:
                    break

            # --- CLEANING LOGIC (Remove Underscores for Output) ---
            # "siêu_thị" -> "siêu thị"
            unique_new_terms_clean = [t.replace("_", " ") for t in selected_terms]

            # Create the expansion string
            expansion_str = " ".join(unique_new_terms_clean)

            # Sanitize expansion string (for safety)
            expansion_str_sanitized = sanitize_text(expansion_str)

            # Save the expansion terms to the CSV
            exp_writer.writerow([doc_id, expansion_str_sanitized])

            # --- TOKENIZATION & TRUNCATION ---
            # Strategy: [Document] + [Expansion]

            # 1. Tokenize the Expansion
            exp_tokens = tokenizer.tokenize(expansion_str)

            # 2. Calculate budget for the Raw Doc
            budget_for_doc = args.max_length - len(exp_tokens)

            # Ensure at least some budget for doc
            if budget_for_doc <= 0:
                # Should rarely happen with a reasonable max_expansion_terms cap
                final_tokens = exp_tokens[:args.max_length]
            else:
                # Tokenize the Raw Doc
                doc_tokens = tokenizer.tokenize(raw_doc_text)

                # Truncate Doc to fit budget
                truncated_doc_tokens = doc_tokens[:budget_for_doc]

                # Combine: [Truncated Raw Doc] + [Expansion]
                final_tokens = truncated_doc_tokens + exp_tokens

            # Decode back to string
            final_text_raw = tokenizer.convert_tokens_to_string(final_tokens)

            # --- FINAL SANITIZATION ---
            # Critical step to ensure TSV compatibility
            final_text_clean = sanitize_text(final_text_raw)

            # Write to TSV
            doc_writer.writerow([doc_id, final_text_clean])
            processed_count += 1

    print(f"Done. Expanded and saved {processed_count} documents.")

if __name__ == "__main__":
    main()
