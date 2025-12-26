import csv
import argparse
import sys
from tqdm import tqdm

# Increase CSV field size limit to handle large document texts
csv.field_size_limit(sys.maxsize)

def load_doc_mapping(doc_mapping_path):
    """
    Loads unique_doc_mapping.csv into a dictionary: Document Text -> Doc ID
    """
    print(f"Loading document mapping from {doc_mapping_path}...")
    doc_map = {}
    with open(doc_mapping_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading Docs"):
            if 'document' in row and 'doc_id' in row:
                # Normalize text by stripping whitespace
                doc_text = row['document'].strip()
                doc_map[doc_text] = row['doc_id']
    return doc_map

def load_vifc_data(vifc_path):
    """
    Loads vifc_test_set.csv into a dictionary: Query Text -> Set of Document Texts
    """
    print(f"Loading VIFC data from {vifc_path}...")
    query_to_docs = {}
    with open(vifc_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading VIFC"):
            if 'query' in row and 'document' in row:
                q_text = row['query'].strip()
                d_text = row['document'].strip()

                if q_text not in query_to_docs:
                    query_to_docs[q_text] = set()
                query_to_docs[q_text].add(d_text)
    return query_to_docs

def main():
    parser = argparse.ArgumentParser(description="Generate test_queries.tsv and qrels.test.tsv")
    
    # Inputs
    parser.add_argument("--test_query_mapping", type=str, required=True, help="Path to test_query_mapping.csv")
    parser.add_argument("--vifc_file", type=str, required=True, help="Path to vifc_test_set.csv")
    parser.add_argument("--doc_mapping", type=str, required=True, help="Path to unique_doc_mapping.csv")

    # Outputs
    parser.add_argument("--output_queries", type=str, required=True, help="Path to output test_queries.tsv")
    parser.add_argument("--output_qrels", type=str, required=True, help="Path to output qrels.test.tsv")

    args = parser.parse_args()

    # 1. Load Mappings
    doc_text_to_id = load_doc_mapping(args.doc_mapping)
    vifc_query_to_docs = load_vifc_data(args.vifc_file)

    print("Processing queries and generating outputs...")

    qrels_count = 0
    queries_count = 0
    missing_docs = 0

    with open(args.test_query_mapping, 'r', encoding='utf-8') as f_in, \
         open(args.output_queries, 'w', encoding='utf-8', newline='') as f_q_out, \
         open(args.output_qrels, 'w', encoding='utf-8', newline='') as f_rel_out:

        reader = csv.DictReader(f_in)

        # We write manually to ensure strict control over tabs/newlines

        for row in tqdm(reader, desc="Processing"):
            if 'query_id' not in row or 'query' not in row:
                continue

            qid = row['query_id'].strip()
            query_text = row['query'].strip()

            # 2. Write to test_queries.tsv
            # Sanitize query text to remove tabs/newlines which break TSV
            clean_query = query_text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
            f_q_out.write(f"{qid}\t{clean_query}\n")
            queries_count += 1

            # 3. Generate Qrels
            # Find associated documents from VIFC dataset
            relevant_docs_texts = vifc_query_to_docs.get(query_text, [])

            for doc_text in relevant_docs_texts:
                # Find the doc_id using the document text
                doc_id = doc_text_to_id.get(doc_text)

                if doc_id:
                    # Write QREL: qid 0 docid 1
                    f_rel_out.write(f"{qid}\t0\t{doc_id}\t1\n")
                    qrels_count += 1
                else:
                    missing_docs += 1

    print("\nProcessing Complete.")
    print(f"Generated {queries_count} queries in {args.output_queries}")
    print(f"Generated {qrels_count} qrels in {args.output_qrels}")
    if missing_docs > 0:
        print(f"Warning: Could not find IDs for {missing_docs} document texts (mapping mismatch).")

if __name__ == "__main__":
    main()
