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


def get_corpus_docnos(path: Path) -> set:
    """Reads the processed collection and returns a set of valid docnos."""
    print(f"Scanning collection for all valid docnos from: {path}")
    docnos = set()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                # Check for valid line and non-empty text
                if len(parts) == 2 and parts[1].strip():
                    docnos.add(parts[0])
    except FileNotFoundError:
        print(f"Error: Processed collection file not found at {path}", file=sys.stderr)
        sys.exit(1)

    if not docnos:
        print("Error: No valid documents found in collection file.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(docnos)} valid documents in the corpus.")
    return docnos


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
    parser.add_argument('--batch_size', type=int, default=None,
                        help="Batch size for evaluation.")
    parser.add_argument('--verbose', action='store_true',
                        help="Enable verbose logging.")
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
    qrels_df = load_qrels_df(processed_qrels_path)
    corpus_docnos = get_corpus_docnos(processed_collection_path)

    # Filter Qrels: Keep only judgments for documents that are in our index
    print(f"Original qrels count: {len(qrels_df)}")
    qrels_df = qrels_df[qrels_df['docno'].isin(corpus_docnos)]
    print(f"Filtered qrels (docno in corpus): {len(qrels_df)}")

    # Filter Queries: Keep only queries that have at least one valid judgment
    print(f"Original queries count: {len(queries_df)}")
    # First, filter for non-empty query text
    queries_df = queries_df[queries_df['query'].fillna('').str.strip() != '']
    print(f"Filtered queries (non-empty text): {len(queries_df)}")

    # Second, filter for queries present in the *filtered* qrels
    valid_qids = set(qrels_df['qid'].unique())
    queries_df = queries_df[queries_df['qid'].isin(valid_qids)]
    print(f"Filtered queries (in valid qrels): {len(queries_df)}")

    if queries_df.empty or qrels_df.empty:
        print("Error: No valid queries or qrels left after filtering. Aborting.", file=sys.stderr)
        sys.exit(1)

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
            verbose=args.verbose
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
        properties={"termpipelines": ""}
    )

    # 6. Evaluate
    print("Running PyTerrier experiment...")
    results = pt.Experiment(
        [bm25],
        queries_df,
        qrels_df,
        eval_metrics=["recip_rank", "recall_10"],
        batch_size=args.batch_size,
        names=["BM25_on_Expanded_Collection_VN"],
        verbose=args.verbose
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
