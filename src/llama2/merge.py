import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.utils.datasets import CollectionParser
from src.utils.defaults import COLLECTION_TYPES
from src.utils.utils import merge

from src.deep_impact.models.original import DeepImpact
from src.utils.defaults import VNCORE_DIR


def merge_collection_and_expansions(collection_path: Path, collection_type: str, queries_path: Path, output: Path):
    # with open(collection_path) as f, open(queries_path) as q, open(output, 'w') as out:
    #     for line, query_list in tqdm(zip(f, q)):
    #         doc_id, doc = CollectionParser.parse(line, collection_type)
    #         query_list = json.loads(query_list)

    #         assert doc_id == query_list['doc_id'], f"Doc id mismatch: {doc_id} != {query_list['doc_id']}"

    #         doc = merge(doc, query_list['query'])
    #         out.write(f'{doc_id}\t{doc}\n')

    # Get total number of queries for tqdm
    q_len = 0
    with open(queries_path, 'r', encoding='utf-8') as q_check:
        q_len = sum(1 for _ in q_check)
    
    if q_len == 0:
        print("Queries file is empty. Nothing to merge.")
        return

    print(f"Merging {q_len} queries from {queries_path} with {collection_path}...")

    with open(collection_path, 'r', encoding='utf-8') as f, \
         open(queries_path, 'r', encoding='utf-8') as q, \
         open(output, 'w', encoding='utf-8') as out:

        # zip() stops when the shortest iterable (q) ends, which is the exact behavior we want.
        for line, query_list_line in tqdm(zip(f, q), total=q_len, desc="Merging documents"):
            doc_id, doc = CollectionParser.parse(line, collection_type)
            query_list = json.loads(query_list_line)

            assert doc_id == query_list['doc_id'], f"Doc id mismatch: {doc_id} != {query_list['doc_id']}"

            # FIXED: Changed 'query' to 'queries' to match generate.py
            doc = merge(doc, query_list['queries'])
            out.write(f'{doc_id}\t{doc}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collection with generated queries')
    parser.add_argument('--collection_path', type=Path)
    parser.add_argument('--collection_type', type=str, choices=COLLECTION_TYPES)
    parser.add_argument("--vncorenlp_path", type=Path, default=VNCORE_DIR, 
                        help="Path to VnCoreNLP model folder (for Vietnamese processing)")
    parser.add_argument('--queries_path', type=Path)
    parser.add_argument('--output_path', type=Path)
    args = parser.parse_args()

    DeepImpact._vncorenlp_path = args.vncorenlp_path

    merge_collection_and_expansions(args.collection_path, args.collection_type, args.queries_path, args.output_path)
