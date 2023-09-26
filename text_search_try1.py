
import numpy as np
from rank_bm25 import BM25L, BM25Okapi, BM25Plus


def softmax(scores):
    exp_scores = np.exp(scores - np.max(scores))
    return exp_scores / exp_scores.sum(axis=0)

def bm25_search(keyword: str, text: str) -> list[tuple[str, float]]:
    """Search for the n-gram query in the text using the BM25 algorithm."""
    lines = [line for line in text.split("\n") if line.strip()]
    words = [line.split() for line in lines]
    print("BM25 words:", words)
    print("BM25 keyword:", keyword.split())
    bm25 = BM25L(words, k1=1.5, b=0.75, delta=0.5) 
    # print("Corpus size:", bm25.corpus_size)
    # print("Average document length:", bm25.avgdl)
    # print("BM25 doc_freq:", bm25.doc_freqs)
    # print("BM25 idf:", bm25.idf)
    # print("BM25 doc_len:", bm25.doc_len)
    bm25_scores = bm25.get_scores(keyword.split())
    print("BM25 search scores:", bm25_scores)
    normalized_scores = softmax(bm25_scores)
    print("BM25 normalize scores:", normalized_scores)
    return list(zip(lines, normalized_scores))


def search(text: str, query: str) -> list[tuple[str, int]]:
    """Search for the query in the text using the BM25 algorithm."""
    # Search for the query in the text using the BM25 algorithm
    bm25_results = bm25_search(query, text)
    
    # Sort the results by the BM25 score
    bm25_results.sort(key=lambda x: x[1], reverse=True)
    
    # Return the results
    return bm25_results

def main():
    sample_text = """
    Fuzzy searches help in finding words that are close in spelling. Semantic searches is more complex.
    It is used to searches words that are similar in meaning.
    A search engine is a software system that is designed to carry out web searches and Not all search engines are created equal. And not all search engines are created equal.
    We are trying to find exact and inexact matches.
    At the end of the day, we are trying to find the most relevant results.
    This is the end of the sample text.
    """
    query = "searches for"
    results = bm25_search(query, sample_text)
    print(results)

if __name__ == '__main__':
    main()