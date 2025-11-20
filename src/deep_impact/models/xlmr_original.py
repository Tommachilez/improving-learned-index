import os
import string
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Set

import numpy as np
import tokenizers
import torch
import torch.nn as nn
# from transformers import BertPreTrainedModel, BertModel
from transformers import XLMRobertaPreTrainedModel, XLMRobertaModel, AutoTokenizer
from underthesea import text_normalize

from src.utils.checkpoint import ModelCheckpoint


class DeepImpact(XLMRobertaPreTrainedModel):
    max_length = 512
    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
    punctuation = set(string.punctuation)

    def __init__(self, config):
        super(DeepImpact, self).__init__(config)
        self.bert = XLMRobertaModel(config)
        self.impact_score_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, 1),
            nn.ReLU()
        )
        self.init_weights()

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param input_ids: Batch of input ids
        :param attention_mask: Batch of attention masks
        :param token_type_ids: Batch of token type ids
        :return: Batch of impact scores
        """
        bert_output = self._get_bert_output(input_ids, attention_mask, token_type_ids)
        return self._get_term_impact_scores(bert_output.last_hidden_state)

    def _get_bert_output(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
            output_attentions: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        :param input_ids: Batch of input ids
        :param attention_mask: Batch of attention masks
        :param token_type_ids: Batch of token type ids
        :param output_attentions: Whether to output attentions
        :return: Batch of BERT outputs
        """
        return self.bert(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            output_attentions=output_attentions
        )

    def _get_term_impact_scores(
            self,
            last_hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param last_hidden_state: Last hidden state from BERT
        :return: Impact scores
        """
        return self.impact_score_encoder(last_hidden_state)

    @classmethod
    def process_query_and_document(cls, query: str, document: str, max_length: Optional[int] = None) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Process query and document to feed to the model
        :param query: Query string
        :param document: Document string
        :param max_length: Max number of tokens to process
        :return: Tuple: Document Tokens, Mask with 1s corresponding to first tokens of document terms in the query
        """
        query_terms = cls.process_query(query)
        encoded, term_to_token_index = cls.process_document(document, max_length=max_length)

        return encoded, cls.get_query_document_token_mask(query_terms, term_to_token_index, max_length)

    @classmethod
    def get_query_document_token_mask(cls, query_terms: Set[str], term_to_token_index: Dict[str, int],
                                      max_length: Optional[int] = None) -> torch.Tensor:
        if max_length is None:
            max_length = cls.max_length

        mask = np.zeros(max_length, dtype=bool)
        token_indices_of_matching_terms = [v for k, v in term_to_token_index.items() if k in query_terms]
        mask[token_indices_of_matching_terms] = True

        return torch.from_numpy(mask)

    @classmethod
    def process_query(cls, query: str) -> Set[str]:
        query = cls.tokenizer.backend_tokenizer.normalizer.normalize_str(query)
        return set(filter(lambda x: x not in cls.punctuation,
                          map(lambda x: x[0], cls.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(query))))

    @classmethod
    def process_document(cls, document: str, max_length: Optional[int] = None) -> Tuple[tokenizers.Encoding, Dict[str, int]]:
        """
        Encodes the document and maps each unique term (non-punctuation) to its corresponding first token's index.
        :param document: Document string
        :return: Tuple: Encoded document, Dict mapping unique non-punctuation document terms to first token index
        """

        if max_length is None:
            max_length = cls.max_length

        document = cls.tokenizer.backend_tokenizer.normalizer.normalize_str(document)
        document_terms = [x[0] for x in cls.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(document)]
        
        encoded = cls.tokenizer.encode_plus(
            document_terms,
            is_split_into_words=True,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors=None  # Get lists
        )

        class MockEncoding:
            def __init__(self, encoding_dict, tokenizer):
                self.ids = encoding_dict['input_ids']
                self.attention_mask = encoding_dict['attention_mask']
                # RoBERTa doesn't use type_ids, but the model code expects them.
                # We'll create a list of zeros.
                self.type_ids = [0] * len(self.ids)
                self.tokens = tokenizer.convert_ids_to_tokens(self.ids)

        # Adapt encoded to AutoTokenizer to have the same interface as tokenizers.Encoding
        mock_encoded = MockEncoding(encoded, cls.tokenizer)

        term_index_to_token_index = {}
        counter = 0
        # mock_encoded.tokens[1:] because the first token is '<s>' (RoBERTa's CLS)
        for i, token in enumerate(mock_encoded.tokens[1:], start=1):
            if token == '</s>':  # Stop at the end token
                break

            # For SentencePiece (XLM-R), new tokens start with a ' ' (U+2581)
            if token.startswith(" ") or (i == 1):
                term_index_to_token_index[counter] = i
                counter += 1
                print("Term to token:", term_index_to_token_index)

        # filter out duplicate terms, punctuations, and terms whose tokens overflow
        filtered_term_to_token_index = {}
        for i, term in enumerate(document_terms):
            if term not in filtered_term_to_token_index \
                    and term not in cls.punctuation \
                    and i in term_index_to_token_index:
                filtered_term_to_token_index[term] = term_index_to_token_index[i]

        return mock_encoded, filtered_term_to_token_index

    @classmethod
    def load(cls, checkpoint_path: Optional[Union[str, Path]] = None):
        model = cls.from_pretrained('xlm-roberta-base')
        if checkpoint_path is not None:
            if os.path.exists(checkpoint_path):
                ModelCheckpoint.load(model=model, last_checkpoint_path=checkpoint_path)
            else:
                model = cls.from_pretrained(checkpoint_path)

        # Configure tokenizer padding and truncation
        # (AutoTokenizer handles this, but we ensure it matches)
        cls.tokenizer.model_max_length = cls.max_length
        return model

    @staticmethod
    def compute_term_impacts(
            documents_term_to_token_index_map: List[Dict[str, int]],
            outputs: torch.Tensor,
    ) -> List[List[Tuple[str, float]]]:
        """
        Computes the impact scores of each term in each document
        :param documents_term_to_token_index_map: List of dictionaries mapping each unique term to its first token index
        :param outputs: Batch of model outputs
        :return: Batch of lists of tuples of document terms and their impact scores
        """
        impact_scores = outputs.squeeze(-1).cpu().numpy()

        term_impacts = []
        for i, term_to_token_index_map in enumerate(documents_term_to_token_index_map):
            term_impacts.append([
                (term, impact_scores[i][token_index])
                for term, token_index in term_to_token_index_map.items()
            ])

        return term_impacts

    def get_impact_scores(self, document: str) -> List[Tuple[str, float]]:
        """
        Get impact scores for each term in the document
        :param document: Document string
        :return: List of tuples of document terms and their impact scores
        """
        encoded, term_to_token_index = self.process_document(document)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([encoded.attention_mask], dtype=torch.long).to(self.device)
        token_type_ids = torch.tensor([encoded.type_ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self(input_ids, attention_mask, token_type_ids)

        return self.compute_term_impacts([term_to_token_index], outputs)[0]

    def get_impact_scores_batch(self, documents: List[str]) -> List[List[Tuple[str, float]]]:
        """
        Get impact scores for each term in each document in batches
        :param documents: List of document strings
        :return: List of lists of tuples of document terms and their impact scores
        """
        # Process all documents in batch
        encoded_docs = []
        term_to_token_maps = []
        for doc in documents:
            encoded, term_map = self.process_document(doc)
            encoded_docs.append(encoded)
            term_to_token_maps.append(term_map)

        # Create batched tensors
        input_ids = torch.tensor([enc.ids for enc in encoded_docs], dtype=torch.long).to(self.device)
        attention_mask = torch.tensor([enc.attention_mask for enc in encoded_docs], dtype=torch.long).to(self.device)
        token_type_ids = torch.tensor([enc.type_ids for enc in encoded_docs], dtype=torch.long).to(self.device)

        # Get model outputs for full batch
        with torch.no_grad():
            outputs = self(input_ids, attention_mask, token_type_ids)

        # Compute impact scores for all documents
        return self.compute_term_impacts(term_to_token_maps, outputs)
