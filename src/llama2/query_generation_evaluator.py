import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import pyterrier as pt
from tqdm import tqdm


def load_expanded_documents(expanded_collection_path: Path):
    """
    Loads an expanded collection TSV file {docno, text} as a
    generator of dictionaries for pt.IterDictIndexer.
    """
    print(f"Loading expanded documents from: {expanded_collection_path}")
    try:
        with open(expanded_collection_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    docno, text = parts
                    yield {'docno': docno, 'text': text}
                else:
                    print(f"Skipping malformed line: {line.strip()}")

    except FileNotFoundError:
        print(f"Error: The file {expanded_collection_path} was not found.")
        sys.exit(1)


def load_queries_df(queries_path: Path) -> pd.DataFrame:
    """Loads dev queries (qid, query) into a DataFrame for PyTerrier."""
    print(f"Loading queries from: {queries_path}")
    df = pd.read_csv(
        queries_path,
        sep='\t',
        header=None,
        names=['qid', 'query'],
        dtype=str
    )
    return df


def load_qrels_df(qrels_path: Path) -> pd.DataFrame:
    """Loads dev qrels (qid, _, docno, label) into a DataFrame for PyTerrier."""
    print(f"Loading qrels from: {qrels_path}")
    df = pd.read_csv(
        qrels_path,
        sep='\t',
        header=None,
        names=['qid', 'unused', 'docno', 'label'],
        dtype={'qid': str, 'docno': str, 'label': int} # Label must be int
    )
    # Return only the columns PyTerrier needs
    return df[['qid', 'docno', 'label']]


def count_lines(filepath: Path) -> int:
    """Helper function to count lines for tqdm."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a pre-expanded collection using BM25 as per the DeeperImpact paper (Sec 3.3)."
    )
    parser.add_argument('--expanded_collection_path', type=Path, required=True,
                        help="Path to the pre-expanded collection file (TSV: docno, text).")
    parser.add_argument('--queries_path', type=Path, required=True,
                        help="Path to the evaluation queries file (e.g., queries.dev.tsv).")
    parser.add_argument('--qrels_path', type=Path, required=True,
                        help="Path to the evaluation qrels file (e.g., qrels.dev.tsv).")
    parser.add_argument('--output_dir', type=Path, required=True,
                        help="Directory to store the generated index and the final results.csv.")

    args = parser.parse_args()

    # --- 0. Setup ---
    if not pt.java.started():
        pt.java.init()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Query/Qrels Data (as DataFrames) ---
    queries_df = load_queries_df(args.queries_path)
    qrels_df = load_qrels_df(args.qrels_path)

    # --- 2. Create Index ---
    index_path_str = str(args.output_dir / "evaluation_index")

    if not os.path.exists(index_path_str + "/data.properties"):
        print(f"Index not found. Creating index at: {index_path_str}")

        # Create the generator for the documents
        doc_generator = load_expanded_documents(args.expanded_collection_path)

        # Get total number of docs for progress bar
        total_docs = count_lines(args.expanded_collection_path)

        indexer = pt.IterDictIndexer(
            index_path_str,
            stemmer=None,  # Default: EnglishStemmer
            stopwords=None,  # Default: English stopwords
            overwrite=True,
            verbose=True
        )

        # Wrap generator in tqdm for progress
        index_ref = indexer.index(
            tqdm(doc_generator, total=total_docs, desc="Indexing documents"),
            meta=['docno', 'text']
        )
        print(f"Index created at: {index_ref.toString()}")
    else:
        print(f"Loading existing index from: {index_path_str}")

    # Load the index
    index = pt.IndexFactory.of(index_path_str)
    print(f"Index loaded: {index.getCollectionStatistics().toString()}")

    # --- 3. Run BM25 Retrieval ---
    print("Running BM25 retrieval...")
    bm25 = pt.terrier.Retriever(index, wmodel="BM25")

    # --- 4. Evaluate ---
    print("Running PyTerrier experiment...")
    results = pt.Experiment(
        [bm25],
        queries_df,
        qrels_df,
        eval_metrics=["recip_rank_10", "recall_1000"],
        names=["BM25_on_Expanded_Collection"]
    )

    # --- 5. Consolidate and Save Results ---
    print("\n--- Evaluation Complete ---")
    final_results_df = results.set_index('name')
    print(final_results_df)

    output_csv_path = args.output_dir / "bm25_evaluation_results.csv"
    final_results_df.to_csv(output_csv_path)
    print(f"\nResults saved to {output_csv_path}")

if __name__ == "__main__":
    main()
