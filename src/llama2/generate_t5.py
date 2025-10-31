import json
import re
from pathlib import Path
from typing import List, Optional
import argparse # Added import

import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.utils.datasets import CollectionParser
from src.utils.defaults import (
    DEVICE,
    COLLECTION_TYPES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NUM_RETURN_SEQUENCES,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P
)

class T5QueryGenerator:
    """
    Generates queries for documents using a T5-based model.
    """
    def __init__(self, model_name_or_path: str, max_tokens: int):
        """
        Initializes the T5 query generator.

        Args:
            model_name_or_path: Path or Hugging Face ID of the T5 model.
            max_tokens: Maximum input sequence length for the tokenizer.
        """
        self.max_tokens = max_tokens
        # Load the specified T5 model and tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.model.to(DEVICE)
        self.model.eval()
        print(f"T5 Query Generator initialized with model: {model_name_or_path}")

    @torch.no_grad()
    def generate(self, documents: List[str], **kwargs) -> List[List[str]]:
        """
        Generates queries for a batch of documents.

        Args:
            documents: A list of document texts.
            **kwargs: Additional arguments passed to the model's generate method
                      (e.g., num_return_sequences, max_new_tokens).

        Returns:
            A list where each element is a list of generated queries for the corresponding document.
        """
        if 'num_return_sequences' not in kwargs:
             raise ValueError("`num_return_sequences` must be provided in generate kwargs")
        n_ret_seq = kwargs['num_return_sequences']

        # Tokenize the input documents
        inputs = self.tokenizer.batch_encode_plus(
            documents,
            return_tensors='pt',
            padding=True,
            max_length=self.max_tokens,
            truncation=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()} # Move tensors to the configured device

        # Generate outputs from the model
        outputs = self.model.generate(**inputs, **kwargs)

        # Decode the generated token IDs into text
        predicted_queries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Group queries by the original document
        grouped_queries = [predicted_queries[i: i + n_ret_seq] for i in range(0, len(predicted_queries), n_ret_seq)]
        return grouped_queries

def generate_queries_and_save(arguments, query_generator, doc_batch, doc_ids):
    """
    Generates queries for a batch of documents and saves them to the output file.
    """
    queries_list = query_generator.generate(
        doc_batch,
        num_return_sequences=arguments.num_return_sequences,
        max_new_tokens=arguments.max_new_tokens,
        do_sample=True, # Using sampling parameters from original script
        top_k=arguments.top_k,
        top_p=arguments.top_p
    )

    # Append generated queries to the output JSONL file
    with open(arguments.output_path, 'a', encoding='utf-8') as out:
        for i, queries in enumerate(queries_list):
            json.dump({'doc_id': doc_ids[i], 'queries': queries}, out)
            out.write('\n')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate queries from documents using a T5 model.')

    # --- T5 Specific Arguments ---
    parser.add_argument('--t5_model_path', type=str, default='doc2query/msmarco-vietnamese-mt5-base-v1',
                        help="Path or Hugging Face ID of the T5 model for query generation.")

    # --- Common Arguments (from original script) ---
    parser.add_argument('--collection_path', type=Path, required=True,
                        help="Path to the document collection file (e.g., collection.tsv).")
    parser.add_argument('--collection_type', type=str, choices=COLLECTION_TYPES, required=True,
                        help="Format of the collection file (e.g., 'msmarco', 'beir').")
    parser.add_argument('--output_path', type=Path, required=True,
                        help="Path to save the output JSONL file containing doc_id and generated queries.")
    parser.add_argument('--batch_size', type=int, default=16, # Adjusted default batch size for potentially larger T5 models
                        help="Number of documents to process in each batch.")
    parser.add_argument('--num_return_sequences', type=int, default=DEFAULT_NUM_RETURN_SEQUENCES,
                        help="Number of queries to generate per document.")
    parser.add_argument('--max_new_tokens', type=int, default=DEFAULT_MAX_NEW_TOKENS,
                        help="Maximum number of new tokens to generate for each query.")
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_MAX_TOKENS,
                        help="Maximum input sequence length for the tokenizer.")
    parser.add_argument('--top_k', type=int, default=DEFAULT_TOP_K,
                         help="Top-k sampling parameter for generation.")
    parser.add_argument('--top_p', type=float, default=DEFAULT_TOP_P,
                         help="Top-p (nucleus) sampling parameter for generation.")

    # --- NEW: Arguments from generate.py ---
    parser.add_argument('--num_doc', type=int, default=None,
                        help='The total number of documents to process from the collection.')
    parser.add_argument('--continue_processing', action='store_true',
                        help='Continue processing from the last saved doc. Requires --output_path to exist.')

    args = parser.parse_args()

    # --- NEW: Logic for --continue_processing ---
    docs_to_skip = 0
    if args.continue_processing:
        if args.output_path.exists():
            print(f"Resuming processing. Counting processed documents in {args.output_path}...")
            with open(args.output_path, 'r', encoding='utf-8') as out_f:
                # Count lines efficiently
                docs_to_skip = sum(1 for _ in out_f)
            print(f"Found {docs_to_skip} existing documents. Skipping these in the input file.")
        else:
            # Raise an error, since the flag is invalid without the file
            raise FileNotFoundError(
                f"--continue_processing was set, but output file {args.output_path} does not exist."
            )

    # --- Initialize the T5 Generator ---
    generator = T5QueryGenerator(model_name_or_path=args.t5_model_path, max_tokens=args.max_tokens)

    # --- Process the collection in batches ---
    batch = []
    ids = []
    processed_doc_count = 0 # NEW: Counter for --num_doc
    print(f"Starting query generation from collection: {args.collection_path}")
    print(f"Output will be saved to: {args.output_path}")

    # Ensure output file directory exists
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- MODIFIED: Clear the output file only if it exists AND we are not resuming ---
    if args.output_path.exists() and not args.continue_processing:
        args.output_path.unlink()
        print(f"Cleared existing output file (not resuming): {args.output_path}")


    try:
        with open(args.collection_path, 'r', encoding='utf-8') as f:

            # --- NEW: Skip documents if resuming ---
            if docs_to_skip > 0:
                print(f"Skipping {docs_to_skip} documents...")
                for _ in range(docs_to_skip):
                    # Read and discard lines
                    if next(f, None) is None:
                        print("Warning: Reached end of input file while skipping.")
                        break

            # --- MODIFIED: Adjust tqdm total based on --num_doc ---
            print("Starting processing...")
            pbar = tqdm(f,
                        total=args.num_doc,
                        desc="Processing documents",
                        initial=docs_to_skip)

            for line in pbar:
                # --- NEW: Check if we have processed enough documents ---
                if args.num_doc is not None and (docs_to_skip + processed_doc_count) >= args.num_doc:
                    print(f"\nReached --num_doc limit of {args.num_doc} total documents.")
                    break  # Stop processing

                try:
                    # Parse document ID and text using the utility parser
                    doc_id, doc = CollectionParser.parse(line, args.collection_type)
                    batch.append(doc)
                    ids.append(doc_id)
                    processed_doc_count += 1 # NEW: Count each document as it's added

                    # When batch is full, generate queries and save
                    if len(batch) == args.batch_size:
                        generate_queries_and_save(args, generator, batch, ids)
                        batch = [] # Reset batch
                        ids = []   # Reset ids

                except Exception as e:
                    print(f"Error processing line: {line.strip()}. Error: {e}")
                    # Optionally skip the line or handle error differently

            # Process the final partial batch if it exists
            if batch:
                print("Processing final batch...")
                generate_queries_and_save(args, generator, batch, ids)

        print("Query generation completed successfully.")

    except FileNotFoundError:
        print(f"Error: Collection file not found at {args.collection_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
