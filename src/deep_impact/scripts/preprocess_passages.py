import sys
import csv
import argparse
import os
import string
from typing import Set
from tqdm import tqdm
import py_vncorenlp
from underthesea import text_normalize

# ==========================================
# CONFIGURATION (From preprocess_csv.py)
# ==========================================
STOPWORD_WHITELIST = {
    "không", "chẳng", "chả", "chưa", "phi", "vô", "tránh", "đừng", "chớ",
    "và", "hoặc", "nhưng", "tuy", "nếu", "thì", "vì", "do", "bởi", "tại", "nên", 
    "rằng", "là", "của", "thuộc",
    "tại", "ở", "trong", "ngoài", "trên", "dưới", "giữa", "với", "về", "đến",
    "ai", "gì", "nào", "đâu", "khi", "mấy", "bao_nhiêu", "thế_nào", "sao",
    "bị", "được", "do", "bởi"
}

class VietnameseProcessor:
    def __init__(self, vncorenlp_path: str, stopwords_path: str, use_whitelist: bool = False):
        # 1. Init VnCoreNLP
        if not os.path.exists(vncorenlp_path):
            raise FileNotFoundError(f"VnCoreNLP not found at {vncorenlp_path}")

        try:
            self.rdrsegmenter = py_vncorenlp.VnCoreNLP(
                save_dir=vncorenlp_path,
                annotators=["wseg"]
            )
        except Exception as e:
            print(f"Error init VnCoreNLP: {e}")
            sys.exit(1)

        # 2. Init Stopwords
        self.use_whitelist = use_whitelist
        self.stopwords = self._load_stopwords(stopwords_path)
        self.punctuation = set(string.punctuation)

    def _load_stopwords(self, path: str) -> Set[str]:
        sw = set()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    w = line.strip().lower()
                    if w:
                        token = w.replace(' ', '_').replace('-', '_')
                        if self.use_whitelist and token in STOPWORD_WHITELIST:
                            continue
                        sw.add(token)
        return sw

    def process(self, text: str) -> str:
        """Returns a space-separated string of segmented tokens."""
        if not text: return ""
        try:
            text = text_normalize(text.lower())
        except:
            text = text.lower()

        try:
            # Segment
            sents = self.rdrsegmenter.word_segment(text)
            tokens = [t for sent in sents for t in sent.split()]
            # Filter
            valid = [t for t in tokens if t not in self.punctuation and t not in self.stopwords]
            return " ".join(valid)
        except:
            return ""

def main():
    parser = argparse.ArgumentParser(description="Preprocess passage CSV using VnCoreNLP logic.")

    # Input/Output
    parser.add_argument("--input_csv", required=True, help="Path to best_passage.csv (Columns: passage_id, passage_text)")
    parser.add_argument("--output_csv", required=True, help="Path to save preprocessed CSV")

    # Resources
    parser.add_argument("--vncorenlp_path", required=True, help="Directory containing VnCoreNLP models")
    parser.add_argument("--stopwords_path", required=True, help="Path to stopwords text file")

    # Options
    parser.add_argument("--enable_whitelist", action="store_true", help="Prevent whitelisted stopwords from being removed")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    print(f">>> Initializing VietnameseProcessor...")
    processor = VietnameseProcessor(args.vncorenlp_path, args.stopwords_path, use_whitelist=args.enable_whitelist)

    print(f">>> Processing {args.input_csv} -> {args.output_csv}")

    with open(args.input_csv, 'r', encoding='utf-8') as f_in, \
         open(args.output_csv, 'w', encoding='utf-8', newline='') as f_out:

        reader = csv.DictReader(f_in)

        # Verify headers
        if 'passage_text' not in reader.fieldnames:
            print("Error: Input CSV must contain 'passage_text' column.")
            sys.exit(1)

        # Preserve all columns
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()

        count = 0
        for row in tqdm(reader, desc="Tokenizing Passages"):
            raw_text = row.get('passage_text', "")

            if raw_text:
                processed_text = processor.process(raw_text)
                row['passage_text'] = processed_text

            writer.writerow(row)
            count += 1

    print(f">>> Done. Processed {count} passages.")

if __name__ == "__main__":
    main()
