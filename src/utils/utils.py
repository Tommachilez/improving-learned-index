import re
from typing import List
from src.deep_impact.models.original import DeepImpact


def get_unique_query_terms(query_list, passage):
    model_cls = DeepImpact

    # 1. Get query terms using the model's query processor
    query_terms = model_cls.process_query(' '.join(query_list))

    # 2. Get document terms using THE SAME query processor for consistency
    passage_terms = model_cls.process_query(passage)

    return query_terms.difference(passage_terms)


def merge(document: str, queries: List[str]) -> str:
    document = document.replace('\n', ' ')
    unique_query_terms_str = ' '.join(get_unique_query_terms(queries, document))
    document = re.sub(r"\s{2,}", ' ', f'{document} {unique_query_terms_str}')
    return document
