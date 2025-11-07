import sys
import argparse
import string
import re
import shutil
from pathlib import Path
from typing import Union
from tqdm import tqdm

try:
    import py_vncorenlp
    from underthesea import text_normalize
except ImportError as e:
    print(f"Error: Missing dependencies for this script: {e}", file=sys.stderr)
    print("Please ensure 'py_vncorenlp' and 'underthesea' are installed.", file=sys.stderr)
    sys.exit(1)

try:
    from src.utils.defaults import VNCORE_DIR
except ImportError:
    VNCORE_DIR = None


def count_lines(filepath: Path) -> int:
    """Helper function to count lines for tqdm."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        print(f"Error: File not found for line count: {filepath}", file=sys.stderr)
        return 0


class VietnameseTextProcessor:
    """
    Encapsulates VnCoreNLP loading and text processing logic.
    """
    def __init__(self, model_path: Union[str, Path]):
        self.punctuation = set(string.punctuation)
        self.vncorenlp_path = model_path
        self._vncorenlp = None

        if not self.vncorenlp_path:
            print("Error: VnCoreNLP model path is not set.", file=sys.stderr)
            sys.exit(1)

        print(f"Initializing text processor with VnCoreNLP model path: {self.vncorenlp_path}")
        self.get_vncorenlp()

    def get_vncorenlp(self):
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
        rdrsegmenter = self.get_vncorenlp()
        try:
            processed_text = text_normalize(text.lower())
        except Exception:
            processed_text = text.lower()

        processed_text = re.sub(r'[^\w\s]', '', processed_text)

        try:
            segmented_sents = rdrsegmenter.word_segment(processed_text)
        except Exception:
            segmented_sents = []

        query_terms = [term for sent in segmented_sents for term in sent.split(' ')]
        filtered_terms = [term for term in query_terms if term not in self.punctuation and term.strip()]
        return ' '.join(filtered_terms)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Preprocess collection, queries, and qrels using VnCoreNLP."
    )
    parser.add_argument('--expanded_collection_path', type=Path, required=True,
                        help="Path to the pre-expanded collection file (TSV: docno, text).")
    parser.add_argument('--queries_path', type=Path, required=True,
                        help="Path to the evaluation queries file (e.g., queries.dev.tsv).")
    parser.add_argument('--qrels_path', type=Path, required=True,
                        help="Path to the evaluation qrels file (e.g., qrels.dev.tsv).")
    parser.add_argument('--vncorenlp_path', type=Path, default=VNCORE_DIR,
                        help="Path to VnCoreNLP model folder.")
    parser.add_argument('--output_dir', type=Path, required=True,
                        help="Directory to store processed files.")

    args = parser.parse_args()

    if not args.vncorenlp_path:
        print("Error: --vncorenlp_path must be specified or src.utils.defaults.VNCORE_DIR must be importable.", file=sys.stderr)
        sys.exit(1)

    print("--- STAGE 1: PREPROCESSING ---")

    # 1. Initialize Processor
    processor = VietnameseTextProcessor(args.vncorenlp_path)

    # 2. Define output paths
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_collection_path = out_dir / "processed_collection.tsv"
    processed_queries_path = out_dir / "processed_queries.tsv"
    processed_qrels_path = out_dir / "processed_qrels.tsv"

    # 3. Process Collection
    print(f"Processing collection: {args.expanded_collection_path}")
    total_docs = count_lines(args.expanded_collection_path)
    if total_docs > 0:
        with open(args.expanded_collection_path, 'r', encoding='utf-8') as fin, \
             open(processed_collection_path, 'w', encoding='utf-8') as fout:

            for line in tqdm(fin, total=total_docs, desc="Processing documents"):
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    docno, text = parts
                    fout.write(f"{docno}\t{processor.process_text(text)}\n")
    print(f"Processed collection saved to: {processed_collection_path}")

    # 4. Process Queries
    print(f"Processing queries: {args.queries_path}")
    total_queries = count_lines(args.queries_path)
    if total_queries > 0:
        with open(args.queries_path, 'r', encoding='utf-8') as fin, \
             open(processed_queries_path, 'w', encoding='utf-8') as fout:

            for line in tqdm(fin, total=total_queries, desc="Processing queries"):
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    qid, query = parts
                    fout.write(f"{qid}\t{processor.process_text(query)}\n")
    print(f"Processed queries saved to: {processed_queries_path}")

    # 5. Copy Qrels
    print(f"Copying qrels: {args.qrels_path}")
    try:
        shutil.copyfile(args.qrels_path, processed_qrels_path)
        print(f"Qrels copied to: {processed_qrels_path}")
    except Exception as e:
        print(f"Error copying qrels: {e}", file=sys.stderr)

    print("--- STAGE 1: Preprocessing complete. ---")

if __name__ == "__main__":
    main()
