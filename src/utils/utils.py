import re
from typing import List
from underthesea import word_tokenize, text_normalize


def get_term_set(text):
    # 1. Normalize the text (e.g., remove diacritics, standardize punctuation)
    #    Note: text_normalize also lowercases by default.
    normalized_text = text_normalize(text)

    # 2. NEW: Remove special characters and punctuation
    #    This regex keeps only word characters (including Vietnamese accented chars)
    #    and whitespace, replacing all other characters with a space.
    cleaned_text = re.sub(r'[^\w\s]', ' ', normalized_text)

    # 3. Tokenize the normalized text.
    #    word_tokenize returns a list of tokens (words and compound phrases).
    tokens = word_tokenize(cleaned_text, format="text").split()
    # return set(re.sub(r'[^\w\s]', ' ', text).lower().split())
    return set(tokens)


def get_unique_query_terms(query_list, passage):
    terms = get_term_set(passage)
    query_terms = get_term_set(' '.join(query_list))
    return query_terms.difference(terms)


def merge(document: str, queries: List[str]) -> str:
    document = document.replace('\n', ' ')
    unique_query_terms_str = ' '.join(get_unique_query_terms(queries, document))
    document = re.sub(r"\s{2,}", ' ', f'{document} {unique_query_terms_str}')
    return document
