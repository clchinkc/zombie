from __future__ import annotations

import re

import chromadb
import torch
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util
from transformers import BertModel, BertTokenizer

# Sample text for demonstration
sample_text = """
Fuzzy search helps in finding words that are close in spelling. Semantic search is more complex.
Hello, this is a sample text. We are trying to find exact and inexact matches.
This is the end of the sample text.
"""

# setup Chroma in-memory, for easy prototyping. Can add persistence easily
# client = chromadb.Client()
# client = chromadb.EphemeralClient()
# setup Chroma with persistence, for production use. Can also use a remote database
client = chromadb.PersistentClient(path="chroma.db", settings=Settings(allow_reset=True))

# client.reset() # reset the database

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-xlm-r-multilingual-v1")

# Create collection. get_collection, get_or_create_collection, delete_collection also available!
# collection = client.create_collection(name="all-my-documents", embedding_function=sentence_transformer_ef, metadata={"hnsw:space": "cosine"}) # Valid options for hnsw:space are "l2", "ip, "or "cosine". The default is "l2".
# collection = client.get_collection(name="all-my-documents")
collection = client.get_or_create_collection(name="all-my-documents", embedding_function=sentence_transformer_ef, metadata={"hnsw:space": "cosine"})


# Add docs to the collection. Can also update and delete. Row-based API coming soon!
collection.add(
    documents=[
        doc for doc in sample_text.split("\n") if doc
    ], # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
    # embeddings=[get_word_embeddings(doc).tolist() for doc in sample_text.split("\n") if doc], # optional
    metadatas=[
        {"title": "Sample text 1"},
        {"title": "Sample text 2"},
        {"title": "Sample text 3"},
    ], # filter by metadata
    ids=[
        "sample-text-1",
        "sample-text-2",
        "sample-text-3",
    ], # unique for each doc
)

# Search for docs
results = collection.query(
    query_texts=["searches"],
    # query_embeddings=[get_word_embeddings("searches").tolist()], # optional
    n_results=3,
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}, # optional filter
    include=["documents", "distances"], # specify what to return. Default is ["documents", "metadatas", "distances", "ids"]
)

print(results)

def get_word_embeddings(sentence: str) -> torch.Tensor:
    """Get BERT embedding for a given sentence."""
    # Initialize BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
    model = BertModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs['last_hidden_state'][:,0,:]

def get_word_embeddings(sentence: str):
    """Get Sentence Transformer embedding for a given sentence."""
    model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')
    # Available models:
    # paraphrase-xlm-r-multilingual-v1
    # distiluse-base-multilingual-cased-v1
    # paraphrase-multilingual-MiniLM-L12-v2
    # paraphrase-multilingual-mpnet-base-v2
    # msmarco-MiniLM-L6-cos-v5
    # msmarco-MiniLM-L12-cos-v5
    # msmarco-distilbert-cos-v5
    # multi-qa-MiniLM-L6-cos-v1
    # multi-qa-distilbert-cos-v1
    # multi-qa-mpnet-base-cos-v1
    return model.encode(sentence)

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute the cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    return dot_product / (magnitude1 * magnitude2)

def exact_search(keyword: str, text: str) -> list[str]:
    """Search for the exact keyword in the text."""
    return [line for line in text.split("\n") if keyword in line]

def regex_search(keyword: str, text: str) -> list[str]:
    """Search using regular expressions."""
    pattern = re.compile(keyword)
    return [line for line in text.split("\n") if pattern.search(line)]

def fuzzy_search(keyword: str, text: str, threshold: int = 80) -> list[str]:
    """Search for words similar to the keyword using fuzzy matching."""
    lines = text.split("\n")
    matches = []
    for line in lines:
        for word in line.split():
            if fuzz.ratio(keyword, word) >= threshold:
                matches.append(line)
                break
    return matches

def semantic_search(keyword: str, text: str, threshold: float = 0.8) -> list[str]:
    """Search for sentences that are semantically similar to the keyword."""
    keyword_embedding = get_word_embeddings(keyword)
    matches = []
    for line in [line for line in text.split("\n") if line]:
        line_embedding = get_word_embeddings(line)
        # similarity = cosine_similarity(keyword_embedding.numpy().flatten(), line_embedding.numpy().flatten())
        similarity = util.cos_sim(torch.tensor(keyword_embedding), torch.tensor(line_embedding))
        # similarity = util.dot_score(torch.tensor(keyword_embedding), torch.tensor(line_embedding))
        print(similarity)
        if similarity > threshold:
            matches.append(line)
    return matches

def search_text(keyword: str, mode: str, text: str = sample_text) -> list[str]:
    """Unified search function."""
    if mode == "exact":
        return exact_search(keyword, text)
    elif mode == "regex":
        return regex_search(keyword, text)
    elif mode == "fuzzy":
        return fuzzy_search(keyword, text)
    elif mode == "semantic":
        return semantic_search(keyword, text)
    else:
        raise ValueError("Invalid search mode")

# Example usage:
# print(search_text("searches", "semantic"))
