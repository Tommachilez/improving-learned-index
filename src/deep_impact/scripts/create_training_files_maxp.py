import csv
import json
import argparse
from collections import defaultdict, Counter
from tqdm import tqdm

def sanitize_text(text):
    """Cleans text by removing excessive whitespace/newlines."""
    if not text:
        return ""
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
    parser = argparse.ArgumentParser(description="Generate Training Collection from Scored Passages.")

    # Inputs
    parser.add_argument("--best_passages", required=True, help="Path to best_passage_ids.csv from score_viranker.py")
    parser.add_argument("--query_mapping", required=True, help="Path to unique_query_mapping.csv")
    parser.add_argument("--pretokenized_queries", required=True, help="Path to queries_pretokenized.jsonl")

    # Outputs
    parser.add_argument("--output_queries_tsv", required=True, help="Output path for queries TSV")
    parser.add_argument("--output_collection_tsv", required=True, help="Output path for Expanded Collection (pid, text)")
    parser.add_argument("--output_expansion_csv", required=True, help="Output path for Expansion Terms CSV (doc_id, added_terms)")

    # Config
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

    # 3. Build Expanded Collection from Best Passages
    print(f"Building collection from {args.best_passages}...")
    print(f"- Output Collection: {args.output_collection_tsv}")

    # We use a set to track processed PIDs to handle duplicates in best_passages.csv
    processed_pids = set()
    written_count = 0

    with open(args.best_passages, 'r', encoding='utf-8') as f_in, \
         open(args.output_collection_tsv, 'w', encoding='utf-8', newline='') as f_coll_out, \
         open(args.output_expansion_csv, 'w', encoding='utf-8', newline='') as f_exp_out:

        reader = csv.DictReader(f_in)

        # Verify headers exist
        required_headers = ['passage_id', 'passage_text']
        for h in required_headers:
            if h not in reader.fieldnames:
                print(f"Error: {args.best_passages} missing required column: '{h}'")
                exit(1)

        coll_writer = csv.writer(f_coll_out, delimiter='\t')
        exp_writer = csv.writer(f_exp_out)
        exp_writer.writerow(["passage_id", "expansion_terms"])

        for row in tqdm(reader, desc="Expanding Passages"):
            passage_id = row['passage_id'].strip()
            passage_text = row['passage_text']

            if not passage_id or not passage_text:
                continue

            # Deduplication: If we already wrote this passage_id, skip it
            if passage_id in processed_pids:
                continue

            processed_pids.add(passage_id)

            # Extract doc_id from passage_id (assuming format "doc_id#index")
            try:
                if '#' in passage_id:
                    doc_id = passage_id.rsplit('#', 1)[0]
                else:
                    doc_id = passage_id # Fallback if no # found
            except Exception:
                doc_id = passage_id

            # --- PREPARE EXPANSIONS ---
            expansion_str = ""
            term_counts = doc_expansions.get(doc_id)

            if term_counts:
                # Check against the PASSAGE text to avoid redundancy
                # (We filter terms already present in this specific window)
                existing_terms = set(passage_text.split())

                selected_terms = []
                for term, count in term_counts.most_common():
                    if term not in existing_terms:
                        selected_terms.append(term)

                    if len(selected_terms) >= args.max_expansion_terms:
                        break

                unique_new_terms_clean = [t.replace("_", " ") for t in selected_terms]
                expansion_str = " ".join(unique_new_terms_clean)

                if expansion_str:
                    sanitized_exp = sanitize_text(expansion_str)
                    exp_writer.writerow([passage_id, sanitized_exp])

            # --- COMBINE & WRITE ---
            if expansion_str:
                final_text = f"{passage_text} {expansion_str}"
            else:
                final_text = passage_text

            final_text_clean = sanitize_text(final_text)

            # Write: PID \t Text
            coll_writer.writerow([passage_id, final_text_clean])
            written_count += 1

    print(f"Done. Wrote {written_count} unique expanded passages to {args.output_collection_tsv}.")

if __name__ == "__main__":
    main()
