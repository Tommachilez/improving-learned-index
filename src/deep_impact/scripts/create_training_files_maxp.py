import csv
import json
import argparse
from collections import defaultdict, Counter
from tqdm import tqdm
from transformers import AutoTokenizer

def sanitize_text(text):
    """Cleans text by removing excessive whitespace/newlines."""
    if not text:
        return ""
    # split() without arguments splits by any whitespace (including \n, \t, \r)
    return " ".join(text.split()).strip()

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
    parser = argparse.ArgumentParser(description="Generate Training Collection from Scored Passages (MaxP) with Term Expansion.")

    # Inputs
    parser.add_argument("--best_passages", required=True, help="Path to best_passage_ids.csv (contains 'passage_id' and 'passage_text')")
    parser.add_argument("--query_mapping", required=True, help="Path to unique_query_mapping.csv")
    parser.add_argument("--pretokenized_queries", required=True, help="Path to queries_pretokenized.jsonl")

    # Outputs
    parser.add_argument("--output_queries_tsv", required=True, help="Output path for queries TSV")
    parser.add_argument("--output_docs_tsv", required=True, help="Output path for Expanded Documents TSV (Final Model Input)")
    parser.add_argument("--output_expansion_csv", required=True, help="Output path for Expansion Terms CSV (doc_id, added_terms)")

    # Config
    parser.add_argument("--model_name", default="xlm-roberta-base", help="Tokenizer model to use for length calculation.")
    parser.add_argument("--max_length", type=int, default=512, help="Max total tokens (doc + expansion)")
    parser.add_argument("--max_expansion_terms", type=int, default=100, help="Max unique expansion terms to append.")

    args = parser.parse_args()

    # 1. Process Query Mapping
    process_queries_mapping(args.query_mapping, args.output_queries_tsv)

    # 2. Load Expansion Terms (Aggregated by Doc ID)
    print(f"Aggregating query terms from {args.pretokenized_queries}...")
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

            for q in queries_list:
                seg = q.get('query_seg', '')
                if seg:
                    terms = seg.split()
                    doc_expansions[pos_doc_id].update(terms)

    # 3. Initialize Tokenizer
    print(f"Loading tokenizer: {args.model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 4. Build Expanded Collection
    print(f"Building collection from {args.best_passages}...")
    print(f"- Expanded Docs TSV: {args.output_docs_tsv}")
    print(f"- Expansion Terms CSV: {args.output_expansion_csv}")
    print(f"- Max Expansion Terms: {args.max_expansion_terms}")

    processed_pids = set()
    written_count = 0

    with open(args.best_passages, 'r', encoding='utf-8') as f_in, \
         open(args.output_docs_tsv, 'w', encoding='utf-8', newline='') as f_doc_out, \
         open(args.output_expansion_csv, 'w', encoding='utf-8', newline='') as f_exp_out:

        reader = csv.DictReader(f_in)

        # Verify headers exist
        required_headers = ['passage_id', 'passage_text']
        for h in required_headers:
            if h not in reader.fieldnames:
                print(f"Error: {args.best_passages} missing required column: '{h}'")
                return

        doc_writer = csv.writer(f_doc_out, delimiter='\t')
        exp_writer = csv.writer(f_exp_out)
        exp_writer.writerow(["passage_id", "expansion_terms"])

        for row in tqdm(reader, desc="Expanding Passages"):
            passage_id = row['passage_id'].strip()
            passage_text_seg = row['passage_text'] # This likely contains underscores (e.g. "Học_sinh")

            if not passage_id or not passage_text_seg:
                continue

            # Deduplication: If we already wrote this passage_id, skip it
            if passage_id in processed_pids:
                continue
            processed_pids.add(passage_id)

            # Extract doc_id from passage_id (assuming format "doc_id#index")
            try:
                doc_id = passage_id.rsplit('#', 1)[0] if '#' in passage_id else passage_id
            except Exception:
                doc_id = passage_id

            # --- PREPARE DATA ---
            # 1. Terms for Deduplication: Use the segmented text
            existing_terms = set(passage_text_seg.split())

            # 2. Text for Output: Clean underscores to get natural text
            passage_text_clean = passage_text_seg.replace('_', ' ')

            # --- SELECT EXPANSIONS ---
            expansion_str = ""
            term_counts = doc_expansions.get(doc_id)

            if term_counts:
                selected_terms = []
                for term, _ in term_counts.most_common():
                    if term not in existing_terms:
                        selected_terms.append(term)

                    if len(selected_terms) >= args.max_expansion_terms:
                        break

                # Clean underscores in expansions (e.g. "máy_tính" -> "máy tính")
                unique_new_terms_clean = [t.replace("_", " ") for t in selected_terms]
                expansion_str = " ".join(unique_new_terms_clean)

                # Sanitize and write to auxiliary CSV
                if expansion_str:
                    sanitized_exp = sanitize_text(expansion_str)
                    exp_writer.writerow([passage_id, sanitized_exp])

            # --- TOKENIZATION & TRUNCATION ---
            # Strategy: [Truncated Original Doc] + [Full Expansion]

            # 1. Tokenize Expansion
            exp_tokens = tokenizer.tokenize(expansion_str) if expansion_str else []

            # 2. Calculate remaining budget for the original document
            budget_for_doc = args.max_length - len(exp_tokens)

            if budget_for_doc <= 0:
                # If expansion takes up entire budget (rare), prioritize expansion
                final_tokens = exp_tokens[:args.max_length]
            else:
                # Tokenize the Clean Document
                doc_tokens = tokenizer.tokenize(passage_text_clean)

                # Truncate Doc
                truncated_doc_tokens = doc_tokens[:budget_for_doc]

                # Combine
                final_tokens = truncated_doc_tokens + exp_tokens

            # Decode back to string
            final_text_raw = tokenizer.convert_tokens_to_string(final_tokens)

            # Final Sanitize (ensure no tabs/newlines break TSV)
            final_text_clean_out = sanitize_text(final_text_raw)

            # Write: PID \t Text
            doc_writer.writerow([passage_id, final_text_clean_out])
            written_count += 1

    print(f"Done. Wrote {written_count} unique expanded passages to {args.output_docs_tsv}.")

if __name__ == "__main__":
    main()
