import sys
import argparse
import string
import re
import shutil
import pandas as pd
from pathlib import Path
from typing import Union, Optional
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


def _load_and_compile_stopwords(path: Union[str, Path]) -> Optional[re.Pattern]:
    """
    Internal helper function to load stopwords and compile them
    into a single, efficient regex pattern.
    """
    try:
        with open(path, "r", encoding='utf-8') as f:
            # Read, lowercase, and strip whitespace from each stopword
            stop_words = [line.strip().lower() for line in f.read().splitlines() if line.strip()]

        if not stop_words:
            print("Warning: Stopwords file was empty. No stopword removal will be applied.")
            return None

        # 1. Sort by length (longest first)
        stop_words.sort(key=len, reverse=True)

        # 2. Create the pattern: \b(word1|word2|nói riêng)\b
        # \b ensures we match whole words only.
        pattern_str = r"\b(" + "|".join(re.escape(word) for word in stop_words) + r")\b"

        # 3. Compile the regex for maximum performance
        # re.IGNORECASE is good practice, though we already .lower()
        return re.compile(pattern_str, flags=re.IGNORECASE)

    except FileNotFoundError:
        print(f"Warning: Stopwords file not found at {path}. Skipping stopword removal.")
        return None
    except Exception as e:
        print(f"Error loading stopwords: {e}")
        return None


def count_lines(filepath: Path) -> int:
    """Helper function to count lines for tqdm."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        print(f"Error: File not found for line count: {filepath}", file=sys.stderr)
        return 0


def process_large_tsv(
    input_path: Path,
    output_path: Path,
    preprocess_func: callable,
    id_col_name: str,
    text_col_name: str,
    chunk_size: int,
    num_doc: Optional[int] = None
):
    """
    Processes a large TSV file in chunks, supporting resumes.
    - id_col_name: 'docno' or 'qid'
    - text_col_name: 'text' or 'query'
    - num_doc: Total quota of documents to process.
    """
    print(f"Starting batch processing for: {input_path}")
    print(f"  Output file: {output_path}")
    print(f"  Chunk size (save_every): {chunk_size}")

    # 1. Check for existing progress by counting lines in the output file
    lines_processed = 0
    if output_path.exists():
        lines_processed = count_lines(output_path)
        print(f"  Resuming from line {lines_processed + 1}...")

    # Check quota logic
    if num_doc is not None:
        if lines_processed >= num_doc:
            print(f"  Quota of {num_doc} documents already reached. Skipping.")
            return
        print(f"  Processing up to {num_doc} documents (total, including existing).")

    total_lines = count_lines(input_path)
    if total_lines == 0:
        print("Input file is empty or unreadable. Skipping.")
        return

    if num_doc is None and lines_processed >= total_lines:
        print("Output file is already complete. Skipping.")
        return

    # Determine tqdm total
    total_pbar = total_lines
    if num_doc is not None:
        total_pbar = min(total_lines, num_doc)

    # 2. Create a pandas iterator
    reader = pd.read_csv(
        input_path,
        sep='\t',
        header=None,
        dtype=object,
        keep_default_na=False,
        chunksize=chunk_size,
        skiprows=lines_processed,
        names=[id_col_name, text_col_name]  # Assign column names
    )

    # 3. Open output file in 'append' (a) mode
    with open(output_path, 'a', encoding='utf-8') as f_out:

        # 4. Set up TQDM progress bar
        with tqdm(total=total_pbar, desc=f"Processing {input_path.name}", initial=lines_processed) as pbar:

            # 5. Iterate over chunks
            for chunk in reader:
                # Quota check inside loop
                if num_doc is not None:
                    remaining = num_doc - lines_processed
                    if remaining <= 0:
                        break
                    if len(chunk) > remaining:
                        chunk = chunk.iloc[:remaining]

                # Fill NaN just in case
                chunk = chunk.fillna("")

                # Apply the processing function to the text column
                chunk[text_col_name] = chunk[text_col_name].map(preprocess_func)

                # Save the processed chunk to the output file
                chunk.to_csv(
                    f_out,
                    sep='\t',
                    index=False,
                    header=False
                )

                # Update progress bar and counter
                processed_count = len(chunk)
                lines_processed += processed_count
                pbar.update(processed_count)

                if num_doc is not None and lines_processed >= num_doc:
                    break

    print(f"Batch processing complete for: {input_path}")


class VietnameseTextProcessor:
    """
    Encapsulates VnCoreNLP loading and text processing logic.
    """
    def __init__(self, model_path: Union[str, Path], stopwords_path: Union[str, Path] = None):
        self.punctuation = set(string.punctuation)
        self.vncorenlp_path = model_path
        self._vncorenlp = None
        self.stopword_pattern = None

        if not self.vncorenlp_path:
            print("Error: VnCoreNLP model path is not set.", file=sys.stderr)
            sys.exit(1)

        print(f"Initializing text processor with VnCoreNLP model path: {self.vncorenlp_path}")
        self.get_vncorenlp()

        if stopwords_path:
            print(f"Loading stopwords from: {stopwords_path}")
            self.stopword_pattern = _load_and_compile_stopwords(stopwords_path)
        else:
            print("No stopwords path provided. Skipping stopword removal.")

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

        # 1. Normalize
        try:
            processed_text = text_normalize(text.lower())
        except Exception:
            processed_text = text.lower()

        # 2. Remove all punctuation (This also fixes the / and ? errors)
        processed_text = re.sub(r'[^\w\s]', '', processed_text, flags=re.UNICODE)

        # 3. Remove stopwords using the pre-compiled regex
        if self.stopword_pattern:
            processed_text = self.stopword_pattern.sub(" ", processed_text)

        # 4. Clean up extra whitespace
        processed_text = re.sub(r"\s+", " ", processed_text).strip()

        try:
            segmented_sents = rdrsegmenter.word_segment(processed_text)
        except Exception:
            segmented_sents = []

        query_terms = [term for sent in segmented_sents for term in sent.split(' ')]
        filtered_terms = [term for term in query_terms if term.strip()]
        return ' '.join(filtered_terms)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Preprocess collection, queries, and qrels using VnCoreNLP."
    )
    parser.add_argument('--collection_path', type=Path, required=True,
                        help="Path to the pre-expanded collection file (TSV: docno, text).")
    parser.add_argument('--queries_path', type=Path, required=True,
                        help="Path to the evaluation queries file (e.g., queries.dev.tsv).")
    parser.add_argument('--qrels_path', type=Path, required=True,
                        help="Path to the evaluation qrels file (e.g., qrels.dev.tsv).")
    parser.add_argument('--vncorenlp_path', type=Path, default=VNCORE_DIR,
                        help="Path to VnCoreNLP model folder.")
    parser.add_argument('--stopwords_path', type=Path, default=None,
                        help="Path to the Vietnamese stopwords file.")
    parser.add_argument('--output_dir', type=Path, required=True,
                        help="Directory to store processed files.")
    parser.add_argument('--chunk_size', type=int, default=10000,
                        help="Lines to process in one batch.")
    parser.add_argument('--num_doc', type=int, default=None,
                        help="Total number of documents to process from the collection.")
    parser.add_argument('--num_queries', type=int, default=None,
                        help="Total number of queries to process from the queries.")

    args = parser.parse_args()

    if not args.vncorenlp_path:
        print("Error: --vncorenlp_path must be specified or src.utils.defaults.VNCORE_DIR must be importable.", file=sys.stderr)
        sys.exit(1)

    print("--- STAGE 1: PREPROCESSING ---")

    # 1. Initialize Processor
    processor = VietnameseTextProcessor(
        args.vncorenlp_path,
        stopwords_path=args.stopwords_path
    )

    # 2. Define output paths
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_collection_path = out_dir / "processed_collection.tsv"
    processed_queries_path = out_dir / "processed_queries.tsv"
    processed_qrels_path = out_dir / "processed_qrels.tsv"

    # 3. Process Collection
    process_large_tsv(
        input_path=args.collection_path,
        output_path=processed_collection_path,
        preprocess_func=processor.process_text,
        id_col_name="docno",
        text_col_name="text",
        chunk_size=args.chunk_size,
        num_doc=args.num_doc
    )

    # 4. Process Queries
    process_large_tsv(
        input_path=args.queries_path,
        output_path=processed_queries_path,
        preprocess_func=processor.process_text,
        id_col_name="qid",
        text_col_name="query",
        chunk_size=args.chunk_size,
        num_doc=args.num_queries
    )

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
