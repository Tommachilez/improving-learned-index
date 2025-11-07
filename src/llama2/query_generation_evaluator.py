import os
import sys
import argparse
from pathlib import Path
import string
from typing import Union

import pandas as pd
import pyterrier as pt
from tqdm import tqdm
import py_vncorenlp
from underthesea import text_normalize
try:
    from src.utils.defaults import VNCORE_DIR 
except ImportError:
    print("Warning: Could not import VNCORE_DIR.", file=sys.stderr)
    VNCORE_DIR = None


class VietnameseTextProcessor:
    """
    Encapsulates VnCoreNLP loading and text processing logic
    to avoid global variables and improve maintainability.
    """
    def __init__(self, model_path: Union[str, Path]):
        self.punctuation = set(string.punctuation)
        self.vncorenlp_path = model_path
        self._vncorenlp = None

        if not self.vncorenlp_path:
            print("Error: VnCoreNLP model path is not set.", file=sys.stderr)
            sys.exit(1)

        print(f"Initializing text processor with VnCoreNLP model path: {self.vncorenlp_path}")
        # Initialize the model immediately on creation
        self.get_vncorenlp()

    def get_vncorenlp(self):
        """Initializes and returns the py_vncorenlp singleton instance."""
        if self._vncorenlp is None:
            save_dir = str(self.vncorenlp_path)

            if not Path(save_dir).exists():
                print(f"Error: VnCoreNLP path not found: {save_dir}", file=sys.stderr)
                print("Please ensure the model is downloaded and the path is correct.", file=sys.stderr)
                sys.exit(1)

            print(f"Loading VnCoreNLP (wseg) from: {save_dir}")
            try:
                self._vncorenlp = py_vncorenlp.VnCoreNLP(
                    save_dir=save_dir,
                    annotators=["wseg"]
                )
            except Exception as e:
                print(f"Error initializing VnCoreNLP: {e}", file=sys.stderr)
                sys.exit(1)

            print("VnCoreNLP loaded successfully.")
        return self._vncorenlp

    def process_text(self, text: str) -> str:
        """
        Processes text (query or document) using VnCoreNLP, normalization,
        and punctuation removal, returning a space-joined string of terms.
        """
        rdrsegmenter = self.get_vncorenlp()

        try:
            processed_text = text_normalize(text.lower())
        except Exception as e:
            print(f"Warning: text_normalize failed for text: '{text[:50]}...'. Error: {e}", file=sys.stderr)
            processed_text = text.lower() # Fallback

        try:
            segmented_sents = rdrsegmenter.word_segment(processed_text)
        except Exception as e:
            if not processed_text.strip():
                segmented_sents = []
            else:
                print(f"Warning: VnCoreNLP error processing text: {e}. Text: '{processed_text[:50]}...'", file=sys.stderr)
                segmented_sents = []

        query_terms = [term for sent in segmented_sents for term in sent.split(' ')]

        filtered_terms = [term for term in query_terms if term not in self.punctuation and term.strip()]
        return ' '.join(filtered_terms)


def load_expanded_documents(expanded_collection_path: Path, processor: VietnameseTextProcessor):
    """
    Loads an expanded collection TSV file {docno, text} as a
    generator of dictionaries for pt.IterDictIndexer.
    
    Applies the provided processor to the document text.
    """
    print(f"Loading and processing expanded documents from: {expanded_collection_path}")
    try:
        total_lines = count_lines(expanded_collection_path)
        if total_lines == 0:
            print(f"Warning: Document file is empty: {expanded_collection_path}", file=sys.stderr)
            return

        with open(expanded_collection_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="Processing documents"):
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    docno, text = parts
                    yield {'docno': docno, 'text': processor.process_text(text)}
                else:
                    print(f"Skipping malformed line: {line.strip()}", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: The file {expanded_collection_path} was not found.")
        sys.exit(1)


def load_queries_df(queries_path: Path, processor: VietnameseTextProcessor) -> pd.DataFrame:
    """Loads dev queries (qid, query) into a DataFrame for PyTerrier."""
    print(f"Loading queries from: {queries_path}")
    df = pd.read_csv(
        queries_path,
        sep='\t',
        header=None,
        names=['qid', 'query'],
        dtype=str
    )
    print("Processing queries with VnCoreNLP...")
    df['query'] = df['query'].apply(processor.process_text)
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
    parser.add_argument('--vncorenlp_path', type=Path, default=VNCORE_DIR,
                        help="Path to VnCoreNLP model folder (for Vietnamese processing)")

    args = parser.parse_args()

    # --- 0. Setup ---
    processor = VietnameseTextProcessor(args.vncorenlp_path)
    if not pt.java.started():
        pt.java.init()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load Query/Qrels Data (as DataFrames) ---
    queries_df = load_queries_df(args.queries_path, processor)
    qrels_df = load_qrels_df(args.qrels_path)

    # --- 2. Create Index ---
    index_path_str = str(args.output_dir / "evaluation_index_vn")

    if not os.path.exists(index_path_str + "/data.properties"):
        print(f"Index not found. Creating index at: {index_path_str}")

        # Create the generator for the documents
        doc_generator = load_expanded_documents(args.expanded_collection_path, processor)

        indexer = pt.IterDictIndexer(
            index_path_str,
            stemmer=None,  # Default: EnglishStemmer
            stopwords=None,  # Default: English stopwords
            overwrite=True,
            verbose=True
        )

        index_ref = indexer.index(
            doc_generator,
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
    # Disable term pipelines to prevent stemming/stopword removal
    bm25 = pt.terrier.Retriever(index, wmodel="BM25", properties={"termpipelines" : ""})

    # --- 4. Evaluate ---
    print("Running PyTerrier experiment...")
    results = pt.Experiment(
        [bm25],
        queries_df,
        qrels_df,
        eval_metrics=["recip_rank", "recall_1000"],
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
