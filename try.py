import re
from functools import lru_cache

import chromadb
import jieba
import nltk
import torch
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from fuzzywuzzy import fuzz
from googletrans import Translator
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25L
from scipy.special import softmax
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from transformers import BertModel, BertTokenizer

# Sample text for demonstration
sample_text = """
Fuzzy searches help in finding words that are close in spelling. Semantic searches is more complex.
It is used to search for words that are similar in meaning.
We are trying to find exact and inexact matches.
This is the end of the sample text.
"""

# Initialize translator
translator = Translator()

client = chromadb.PersistentClient(path="chroma.db", settings=Settings(allow_reset=True))

client.reset() # reset the database

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="distiluse-base-multilingual-cased-v1")

vectorizer = TfidfVectorizer(analyzer='word')

tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
model = BertModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2')

collection = client.get_or_create_collection(name="all-my-documents", embedding_function=sentence_transformer_ef, metadata={"hnsw:space": "cosine"})


# Add docs to the collection. Can also update and delete. Row-based API coming soon!
collection.add(
    documents=[
        doc for doc in sample_text.split("\n") if doc
    ], # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
    # embeddings=[get_word_embeddings(doc).tolist() for doc in sample_text.split("\n") if doc], # optional
    metadatas=[
        {"metadata_field": "metadata_value"} for doc in sample_text.split("\n") if doc
    ], # filter by metadata
    ids=[
        f"sample-text-{i+1}" for i, line in enumerate(sample_text.split("\n")) if line
    ], # unique for each doc
)


# get_ngrams, get_all_ngrams, get_word_embeddings, and word_embedding_similarity functions are defined.



def exact_search(keyword: str, text: str) -> list[tuple[str, float]]:
    """Search for the exact keyword in the text and return score according to number of occurrences."""
    lines = [line for line in text.split("\n") if line.strip()]
    scores = [line.count(keyword) for line in lines]
    print("Exact search scores:", scores)
    return list(zip(lines, scores))


def ngram_search(keyword: str, text: str) -> list[tuple[str, float]]:
    """Search for n-grams in the text and return score according to the number of occurrences and n-gram size."""
    keyword_ngrams = get_all_ngrams(keyword)
    lines = [line for line in text.split("\n") if line.strip()]
    scores = []

    ngram_weights = {n: n /  len(keyword.split()) for n in range(1, len(keyword.split()) + 1)}
    for line in lines:
        line_ngrams = get_all_ngrams(line)
        matched_ngrams = set(keyword_ngrams) & set(line_ngrams)
        if not matched_ngrams:
            scores.append(0)
            continue
        line_score = max(ngram_weights[n] for _, n in matched_ngrams)
        scores.append(line_score)

    print("Ngram search scores:", scores)
    return list(zip(lines, scores))


def regex_search(keyword: str, text: str) -> list[tuple[str, float]]:
    """Search using regular expressions."""
    pattern_str = re.escape(keyword)
    pattern_str = pattern_str.replace(r"\ ", r"\s+")
    pattern_str = pattern_str.replace(r"\-", r"\-?")
    pattern_str = pattern_str.replace(r"\_", r"\_?")
    pattern_str = r'.*' + pattern_str + r'.*'
    
    if any(vowel in pattern_str for vowel in ['a', 'e', 'i', 'o', 'u']):
        pattern_str = re.sub(r"(a|e|i|o|u)", r"\1?", pattern_str)

    pattern = re.compile(pattern_str, re.IGNORECASE)

    lines = [line for line in text.split("\n") if line.strip()]
    scores = [len(pattern.findall(line)) for line in lines]
    print("Regex search scores:", scores)
    return list(zip(lines, scores))


def fuzzy_search(keyword: str, text: str) -> list[tuple[str, float]]:
    """Search for n-grams similar to the keyword using fuzzy matching."""
    keyword_ngrams = get_all_ngrams(keyword)
    lines = [line for line in text.split("\n") if line.strip()]
    scores = []
    
    for line in lines:
        max_ratio = max(fuzz.ratio(key_ngram[0], line) for key_ngram in keyword_ngrams) / 100
        scores.append(max_ratio)
        
    print("Fuzzy search scores:", scores)
    return list(zip(lines, scores))


def tfidf_search(keyword: str, text: str) -> list[tuple[str, float]]:
    """Search for the query in the text using TF-IDF weighted n-grams."""
    lines = [line for line in text.split("\n") if line.strip()]
    vectorizer.set_params(ngram_range=(1, len(keyword.split())))
    line_vectors = vectorizer.fit_transform(lines)
    query_vector = vectorizer.transform([keyword])
    cosine_similarities = linear_kernel(query_vector, line_vectors).flatten()
    print("TF-IDF search scores:", cosine_similarities)
    return list(zip(lines, cosine_similarities))


def bm25_search(keyword: str, text: str) -> list[tuple[str, float]]:
    """Search for the n-gram query in the text using the BM25 algorithm."""
    lines = [line for line in text.split("\n") if line.strip()]
    words = [line.split() for line in lines]
    bm25 = BM25L(words, k1=1.5, b=0.75, delta=0.5)  # default: k1=1.5, b=0.75, delta=0.5 (k1 controls term saturation (higher = slower saturation), b adjusts length normalization (closer to 1 = more penalty for longer docs), and delta diminishes term saturation in longer documents (higher = less saturation))
    raw_scores = bm25.get_scores(keyword.split())
    scores = softmax(raw_scores)
    print("BM25 search scores:", scores)
    return list(zip(lines, scores))


def word_embedding_search(keyword: str, text: str) -> list[tuple[str, float]]:
    """Search for n-grams similar to the keyword using word embeddings."""
    keyword_ngrams = get_all_ngrams(keyword)
    lines = [line for line in text.split("\n") if line.strip()]
    scores = []

    for line in lines:
        max_similarity = max(word_embedding_similarity(get_word_embeddings(key_ngram[0]), get_word_embeddings(line)) for key_ngram in keyword_ngrams)
        scores.append(max_similarity)

    print("Word embedding search scores:", scores)
    return list(zip(lines, scores))


def semantic_search(keyword: str, text: str) -> list[tuple[str, float]]:
    """Search for sentences that are semantically similar to the keyword using chroma"""
    results = collection.query(
        query_texts=keyword,
        # query_embeddings=[get_word_embeddings("searches").tolist()], # optional
        n_results=3,
        # where={"metadata_field": "is_equal_to_this"}, # optional filter
        # where_document={"$contains":"search_string"}, # optional filter
        include=["documents", "distances"], # specify what to return. Default is ["documents", "metadatas", "distances", "ids"]
    )
    # print cosine similarity which is 1 - cosine distance, cosine distance is a list at results["distances"][0]
    scores = [1 - dist for dist in results["distances"][0]]
    print("Semantic score:", scores)
    return list(zip(results["documents"][0], scores))



# Segmentation, cleaning, preprocessing, and translation functions are defined.


def retrieve_matches(preprocessed_keywords, preprocessed_lines, search_methods):
    """
    Retrieves matches for each keyword using the specified search methods.
    """
    retrieved_matches = {lang: [] for lang in preprocessed_keywords}
    
    for lang, kw in preprocessed_keywords.items():
        for method, search_function in search_methods.items():
            if search_function:
                matches = search_function(kw, "\n".join(preprocessed_lines))
                # Add method information to the match if it's not already included
                matches_with_method = [(matched_text, score, method) for matched_text, score in matches]
                retrieved_matches[lang].extend(matches_with_method)
    
    return retrieved_matches


def rank_matches(retrieved_matches, cleaned_to_original_mapping, weights):
    """
    Ranks the retrieved matches by applying the weights and accumulating the scores.
    """
    scores = {}

    for lang, matches in retrieved_matches.items():
        for match in matches:
            matched_text, score, method = match  # Assuming this is the structure of the matches
            if method not in weights:
                raise ValueError(f"Method {method} not found in weights dictionary.")
            original_text = cleaned_to_original_mapping.get(matched_text, matched_text)
            weighted_score = score * weights[method]
            scores[original_text] = scores.get(original_text, 0) + weighted_score

    # Sort results by the accumulated scores
    ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_results



def search_and_rank(keyword, text=sample_text, preprocess=True, weights={'exact': 0.9, 'ngram': 0.6, 'regex': 0.7, 'fuzzy': 0.8, 'tfidf': 0.7, 'bm25': 0.7, 'word_embedding': 0.8, 'semantic': 1.0}):

    # Split the text once
    lines = [line for line in text.split("\n") if line.strip()]
    
    # Get translations of the keyword using translate_multilanguage
    original_keyword, english_keyword, traditional_keyword, simplified_keyword = translate_multilanguage(keyword)
    
    translations = {
        'original': original_keyword,
        'english': english_keyword,
        'traditional': traditional_keyword,
        'simplified': simplified_keyword
    }

    for lang, trans_keyword in translations.items():
        print(f"{lang.capitalize()} keyword:", trans_keyword)

    # Clean the translated keywords
    cleaned_translations = {lang: clean_text(kw) for lang, kw in translations.items()}
    
    for lang, kw in cleaned_translations.items():
        print(f"Cleaned {lang.capitalize()} keyword:", kw)
    
    cleaned_lines = [clean_text(line) for line in lines]

    if preprocess:
        preprocessed_lines = [preprocess_text(line) for line in cleaned_lines]
        preprocessed_keywords = {lang: preprocess_text(kw) for lang, kw in cleaned_translations.items()}
        for lang, kw in preprocessed_keywords.items():
            print(f"Preprocessed {lang.capitalize()} keyword:", kw)
    else:
        preprocessed_lines = cleaned_lines
        preprocessed_keywords = cleaned_translations
    
    # Create a mapping between processed and original text
    cleaned_to_original_mapping = dict(zip(preprocessed_lines, lines))

    scores = {}

    for lang, kw in preprocessed_keywords.items():
        search_methods = {
            'exact': exact_search if 'exact' in weights else None,
            'regex': regex_search if 'regex' in weights else None,
            'fuzzy': fuzzy_search if 'fuzzy' in weights else None,
            'ngram': ngram_search if 'ngram' in weights else None,
            'tfidf': tfidf_search if 'tfidf' in weights else None,
            'bm25': bm25_search if 'bm25' in weights else None,
            'word_embedding': word_embedding_search if 'word_embedding' in weights else None,
            'semantic': semantic_search if 'semantic' in weights else None,
        }

    # Retrieve matches
    retrieved_matches = retrieve_matches(preprocessed_keywords, preprocessed_lines, search_methods)
    
    # Rank the matches
    ranked_results = rank_matches(retrieved_matches, cleaned_to_original_mapping, weights)
    
    return ranked_results




default_weights = {
    'exact': 0.9,          # Full weight for precise matches.
    #'regex': 0.7,          # Flexible search with moderate precision.
    'fuzzy': 0.8,          # Useful for variations in spellings.
    'ngram': 0.6,          # Useful for broader matches.
    #'tfidf': 0.7,          # Weighs importance of words in a dataset.
    'bm25': 0.7,           # Weighs importance of words in a dataset.
    'word_embedding': 0.8, # Finds semantically similar terms.
    #'semantic': 1.0        # High weight for context and meaning.
}


results = search_and_rank("搜尋 for", sample_text, preprocess=True, weights=default_weights)

print("Results:")
for line, score in results:
    print(f"Line: {line:<{max(len(line) for line, _ in results)}} | Score: {score:>7.3f}")
