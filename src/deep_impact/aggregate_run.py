import argparse
from collections import defaultdict
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_file", required=True, help="The raw output from rank.py (integers)")
    parser.add_argument("--mapping", required=True, help="The pid_mapping.txt file")
    parser.add_argument("--output", required=True, help="The final run file for evaluation")
    parser.add_argument("--top_k", type=int, default=1000)
    args = parser.parse_args()

    # 1. Load Mapping (Index -> Real ID)
    print("Loading ID mapping...")
    index_to_real_id = {}
    with open(args.mapping, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            index_to_real_id[str(idx)] = line.strip()

    # 2. Process Run File
    print("Aggregating results...")
    # Map: qid -> { real_doc_id -> max_score }
    results = defaultdict(lambda: defaultdict(float))

    with open(args.run_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            parts = line.strip().split('\t')
            if len(parts) < 4: continue

            qid = parts[0]
            int_pid = parts[1] # The integer from DeeperImpact
            score = float(parts[3])

            # Translate Integer -> Real Passage ID (e.g., "doc123#0")
            if int_pid not in index_to_real_id:
                continue
            real_passage_id = index_to_real_id[int_pid]

            # Recover Real Doc ID (e.g., "doc123")
            if '#' in real_passage_id:
                real_doc_id = real_passage_id.split('#')[0]
            else:
                real_doc_id = real_passage_id

            # MaxP Logic: Keep highest score for this doc
            if score > results[qid][real_doc_id]:
                results[qid][real_doc_id] = score

    # 3. Write Final Output
    print(f"Writing to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for qid in sorted(results.keys(), key=lambda x: int(x) if x.isdigit() else x):
            # Sort docs by score
            sorted_docs = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:args.top_k]

            for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
                # Standard TREC-style 4-column format for the evaluator
                f.write(f"{qid}\t{doc_id}\t{rank}\t{score:.6f}\n")

if __name__ == "__main__":
    main()
