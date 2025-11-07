import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

try:
    import pyterrier as pt
    import pandas as pd
except ImportError as e:
    print(f"Error: Missing dependencies for this script: {e}", file=sys.stderr)
    print("Please ensure 'python-terrier' and 'pandas' are installed.", file=sys.stderr)
    sys.exit(1)


def count_lines(filepath: Path) -> int:
    """Helper function to count lines for tqdm."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        print(f"Error: File not found for line count: {filepath}", file=sys.stderr)
        return 0


def load_processed_documents(path: Path):
    """Loads the pre-processed collection.tsv for the indexer."""
    print(f"Loading processed documents from: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    # 1. Strip whitespace from the text
                    docno = parts[0]
                    text = parts[1].strip()
                    # 2. Only yield if the text is not empty
                    if text:
                        yield {'docno': docno, 'text': text}
    except FileNotFoundError:
        print(f"Error: Processed collection file not found at {path}", file=sys.stderr)
        print("Please run the 'preprocess' script first.")
        sys.exit(1)


def load_processed_queries(path: Path) -> pd.DataFrame:
    """Loads the pre-processed queries.tsv into a DataFrame."""
    print(f"Loading processed queries from: {path}")
    try:
        return pd.read_csv(
            path, sep='\t', header=None, names=['qid', 'query'], dtype=str
        )
    except FileNotFoundError:
        print(f"Error: Processed queries file not found at {path}", file=sys.stderr)
        print("Please run the 'preprocess' script first.")
        sys.exit(1)


def load_qrels_df(path: Path) -> pd.DataFrame:
    """Loads the qrels file into a DataFrame."""
    print(f"Loading qrels from: {path}")
    try:
        df = pd.read_csv(
            path, sep='\t', header=None,
            names=['qid', 'unused', 'docno', 'label'],
            dtype={'qid': str, 'docno': str, 'label': int}
        )
        return df[['qid', 'docno', 'label']]
    except FileNotFoundError:
        print(f"Error: Processed qrels file not found at {path}", file=sys.stderr)
        print("Please run the 'preprocess' script first.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Run PyTerrier BM25 evaluation on pre-processed data."
    )
    parser.add_argument('--output_dir', type=Path, required=True,
                        help="Directory containing the processed files from Stage 1.")
    args = parser.parse_args()

    print("--- STAGE 2: EVALUATION ---")

    # 1. Start Java VM
    if not pt.java.started():
        pt.java.init()

    # 2. Define paths from output_dir
    out_dir = args.output_dir
    processed_collection_path = out_dir / "processed_collection.tsv"
    processed_queries_path = out_dir / "processed_queries.tsv"
    processed_qrels_path = out_dir / "processed_qrels.tsv"
    index_path_str = str(out_dir / "evaluation_index_vn")

    # 3. Load data
    queries_df = load_processed_queries(processed_queries_path)

    print(f"Loaded {len(queries_df)} queries.")
    # Fill any potential NaN/None values with '', strip whitespace, and check if the query is empty
    queries_df = queries_df[queries_df['query'].fillna('').str.strip() != '']
    print(f"Filtered empty queries. {len(queries_df)} queries remaining.")

    if queries_df.empty:
        print("Error: No valid queries left after filtering. Aborting evaluation.", file=sys.stderr)
        sys.exit(1)

    qrels_df = load_qrels_df(processed_qrels_path)

    # 4. Create Index
    if not os.path.exists(index_path_str + "/data.properties"):
        print(f"Index not found. Creating index at: {index_path_str}")
        doc_generator = load_processed_documents(processed_collection_path)
        total_docs = count_lines(processed_collection_path)

        indexer = pt.IterDictIndexer(
            index_path_str,
            stemmer=None,    # Crucial: Text is already processed
            stopwords=None,  # Crucial: Text is already processed
            overwrite=True,
            verbose=True
        )

        index_ref = indexer.index(tqdm(doc_generator, total=total_docs, desc="Indexing documents"))
        print(f"Index created at: {index_ref.toString()}")
    else:
        print(f"Loading existing index from: {index_path_str}")

    index = pt.IndexFactory.of(index_path_str)
    print(f"Index loaded: {index.getCollectionStatistics().toString()}")

    # 5. Run BM25 Retrieval
    print("Running BM25 retrieval...")
    bm25 = pt.terrier.Retriever(
        index,
        wmodel="BM25",
        properties={"termpipelines": ""} # Crucial: Don't process queries
    )

    # 6. Evaluate
    print("Running PyTerrier experiment...")
    results = pt.Experiment(
        [bm25],
        queries_df,
        qrels_df,
        eval_metrics=["recip_rank", "recall_1000"],
        names=["BM25_on_Expanded_Collection_VN"],
        verbose=True
    )

    # 7. Save Results
    print("\n--- Evaluation Complete ---")
    final_results_df = results.set_index('name')
    print(final_results_df)

    output_csv_path = args.output_dir / "bm25_evaluation_results_vn.csv"
    final_results_df.to_csv(output_csv_path)
    print(f"\nResults saved to {output_csv_path}")
    print("--- STAGE 2: Evaluation complete. ---")

if __name__ == "__main__":
    main()
