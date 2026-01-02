import csv
import argparse
import sys
from tqdm import tqdm

# Increase CSV field size limit for large passage texts
csv.field_size_limit(sys.maxsize)

def main():
    parser = argparse.ArgumentParser(description="Create a unique mapping of Passage ID to Passage Text from best_passages.csv")
    
    parser.add_argument("--input_csv", required=True, help="Path to best_passages.csv (must contain 'passage_id' and 'passage_text')")
    parser.add_argument("--output_csv", required=True, help="Path to output unique_passage_mapping.csv")

    args = parser.parse_args()

    print(f"Processing {args.input_csv} -> {args.output_csv} ...")

    processed_pids = set()
    total_rows = 0
    unique_rows = 0

    with open(args.input_csv, 'r', encoding='utf-8') as f_in, \
         open(args.output_csv, 'w', encoding='utf-8', newline='') as f_out:

        reader = csv.DictReader(f_in)

        # Verify headers
        if 'passage_id' not in reader.fieldnames or 'passage_text' not in reader.fieldnames:
            print(f"Error: Input CSV must contain 'passage_id' and 'passage_text' columns.")
            print(f"Found columns: {reader.fieldnames}")
            sys.exit(1)

        # Initialize writer with only the relevant columns
        fieldnames = ['passage_id', 'passage_text']
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(reader, desc="Filtering Unique Passages"):
            total_rows += 1
            pid = row['passage_id'].strip()

            if not pid:
                continue

            # Deduplicate based on Passage ID
            if pid in processed_pids:
                continue

            processed_pids.add(pid)

            # Write the unique entry
            writer.writerow({
                'passage_id': pid,
                'passage_text': row['passage_text'].strip()
            })
            unique_rows += 1

    print(f"\nProcessing Complete.")
    print(f"Total Rows Read:   {total_rows}")
    print(f"Unique Passages:   {unique_rows}")
    print(f"Output saved to:   {args.output_csv}")

if __name__ == "__main__":
    main()
