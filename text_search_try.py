
import re

import chromadb
import torch
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from fuzzywuzzy import fuzz
from googletrans import Translator
from sentence_transformers import SentenceTransformer, util
from transformers import BertModel, BertTokenizer

# Sample text for demonstration
sample_text = """
Fuzzy searches help in finding words that are close in spelling. Semantic searches is more complex.
It is used to search for words that are similar in meaning.
We are trying to find exact and inexact matches.
This is the end of the sample text.
"""

# setup Chroma in-memory, for easy prototyping. Can add persistence easily
# client = chromadb.Client()
# client = chromadb.EphemeralClient()
# setup Chroma with persistence, for production use. Can also use a remote database
client = chromadb.PersistentClient(path="chroma.db", settings=Settings(allow_reset=True))

# client.reset() # reset the database

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="distiluse-base-multilingual-cased-v1")


# Available models for SentenceTransformer('model_name'), return model.encode(sentence)
# paraphrase-xlm-r-multilingual-v1
# distiluse-base-multilingual-cased-v1
# paraphrase-multilingual-MiniLM-L12-v2
# paraphrase-multilingual-mpnet-base-v2
# msmarco-MiniLM-L6-cos-v5
# msmarco-MiniLM-L12-cos-v5
# msmarco-distilbert-cos-v5
# multi-qa-MiniLM-L6-cos-v1
# sentence-t5-base
# gtr-t5-base
# multi-qa-mpnet-base-cos-v1
# all-mpnet-base-v2
# all-MiniLM-L6-v2
# all-MiniLM-L12-v2
# all-roberta-large-v1
# all-distilroberta-v1

tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
model = BertModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2')


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
        {"metadata_field": "metadata_value"} for doc in sample_text.split("\n") if doc
    ], # filter by metadata
    ids=[
        f"sample-text-{i+1}" for i, line in enumerate(sample_text.split("\n")) if line
    ], # unique for each doc
)


def get_ngrams(text: str, n: int) -> list[str]:
    """Generate n-grams from the text."""
    words = text.split()
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

def get_all_ngrams(text: str) -> list[tuple[str, int]]:
    """Generate all n-grams from the text."""
    words = text.split()
    num_words = len(words)
    ngrams = [(gram, n) for n in range(1, num_words) for gram in get_ngrams(text, n)]
    return ngrams

def get_word_embeddings(sentence: str) -> torch.Tensor:
    """Get BERT embedding for a given sentence."""
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs['last_hidden_state'][:, 0, :]

def word_embedding_similarity(word1_embedding: torch.Tensor, word2_embedding: torch.Tensor) -> float:
    similarity = torch.nn.functional.cosine_similarity(word1_embedding, word2_embedding)
    return similarity.item()





def exact_search(keyword: str, text: str) -> list[tuple[str, float]]:
    """Search for the exact keyword in the text and return score according to number of occurrences."""
    lines = [line for line in text.split("\n") if line]
    scores = [1.0 * line.count(keyword) for line in lines]
    print("Exact search scores:", scores)
    return [(line, score) for line, score in zip(lines, scores)]


def ngram_search(keyword: str, text: str) -> list[tuple[str, float]]:
    """Search for n-grams in the text and return score according to the number of occurrences and n-gram size."""

    # Extract all n-grams from the keyword
    keyword_ngrams = get_all_ngrams(keyword)
    
    lines = [line for line in text.split("\n") if line]
    scores = []

    # Define a weight for each n-gram size
    # This can be changed to any other weight assignment logic
    ngram_weights = {n: n*0.5 for n in range(1, 11)}

    for line in lines:
        line_ngrams = get_all_ngrams(line)

        # Calculate the score for this line based on matching n-grams and their weights
        line_score = 0.0
        for k_gram, k_n in keyword_ngrams:
            for l_gram, l_n in line_ngrams:
                if k_gram == l_gram:
                    # Add the score for this n-gram, weighted by its size
                    line_score += ngram_weights.get(k_n, 0)
        
        scores.append(line_score)

    print("Ngram search scores:", scores)
    return [(line, score) for line, score in zip(lines, scores)]


def regex_search(keyword: str, text: str) -> list[tuple[str, float]]:
    """Search using regular expressions and return score according to number of occurrences."""
    pattern_str = re.escape(keyword)
    # Replace spaces with \s+ to match any number of spaces
    pattern_str = pattern_str.replace(r"\ ", r"\s+")
    # Replace hyphens with optional hyphens
    pattern_str = pattern_str.replace(r"\-", r"\-?")
    # Replace underscores with optional underscores
    pattern_str = pattern_str.replace(r"\_", r"\_?")
    # Replace vowels with optional vowels
    if any(vowel in pattern_str for vowel in ['a', 'e', 'i', 'o', 'u']):
        pattern_str = re.sub(r"(a|e|i|o|u)", r"\1u?", pattern_str)
    pattern = re.compile(pattern_str, re.IGNORECASE)
    lines = [line for line in text.split("\n") if line]
    scores = [1.0 * len(pattern.findall(line)) for line in lines]
    print("Regex search scores:", scores)
    return [(line, score) for line, score in zip(lines, scores)]


def fuzzy_search(keyword: str, text: str) -> list[tuple[str, float]]:
    """Search for words similar to the keyword using fuzzy matching."""
    lines = [line for line in text.split("\n") if line]
    scores = [
        max(fuzz.ratio(keyword, word) for word in line.split()) / 100.0
        for line in lines
    ]
    print("Fuzzy search scores:", scores)
    return [(line, score) for line, score in zip(lines, scores)]


def word_embedding_search(keyword: str, text: str) -> list[tuple[str, float]]:
    """Search for words similar to the keyword using word embeddings."""
    line_list = [line for line in text.split("\n") if line]
    keyword_embedding = get_word_embeddings(keyword)

    scores = []
    for line in line_list:
        line_embeddings = [get_word_embeddings(word) for word in line.split()]
        max_similarity = 0
        for idx, word in enumerate(line.split()):
            similarity = word_embedding_similarity(keyword_embedding, line_embeddings[idx])
            max_similarity = max(max_similarity, similarity)
        scores.append(max_similarity)

    print("Word embedding search scores:", scores)
    return [(line, score) for line, score in zip(line_list, scores)]


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


def translate_multilanguage(sentence):
    translator = Translator()
    
    # Detect the language of the sentence
    detection = translator.detect(sentence)
    detected_lang = detection.lang
    
    translations = {
        'en': None,
        'zh-TW': None,
        'zh-CN': None
    }
    
    # Translate the sentence to the missing languages
    if detected_lang == 'en':
        translations['en'] = sentence
        translations['zh-TW'] = translator.translate(sentence, dest='zh-TW').text
        translations['zh-CN'] = translator.translate(sentence, dest='zh-CN').text
    elif detected_lang in ['zh-TW', 'zh-CN']:
        translations['en'] = translator.translate(sentence, dest='en').text
        if detected_lang == 'zh-TW':
            translations['zh-CN'] = translator.translate(sentence, dest='zh-CN').text
            translations['zh-TW'] = sentence
        else:
            translations['zh-TW'] = translator.translate(sentence, dest='zh-TW').text
            translations['zh-CN'] = sentence
    else:
        raise ValueError("Input sentence is neither in English nor Chinese.")

    return translations['en'], translations['zh-TW'], translations['zh-CN']


def search_and_rank(keyword, text=sample_text, weights={'exact': 1, 'ngram': 0.8, 'regex': 0.8, 'fuzzy': 0.6, 'word_embedding': 0.6, 'semantic': 0.4}):
    # Get translations of the keyword into the three languages
    english_keyword, traditional_keyword, simplified_keyword = translate_multilanguage(keyword)
    keywords = [english_keyword, traditional_keyword, simplified_keyword]
    print("English keyword:", english_keyword)
    print("Traditional Chinese keyword:", traditional_keyword)
    print("Simplified Chinese keyword:", simplified_keyword)

    scores = {}

    for kw in keywords:
        search_methods = {
            'exact': exact_search(kw, text),
            'ngram': ngram_search(kw, text),
            'regex': regex_search(kw, text),
            'fuzzy': fuzzy_search(kw, text),
            'word_embedding': word_embedding_search(kw, text),
            'semantic': semantic_search(kw, text)
        }

        for method, matches in search_methods.items():
            if not matches:  # Handle case where no matches are found
                continue

            max_score = max([score for _, score in matches])
            normalized_matches = {text: score / (max_score + 1e-6) for text, score in matches}

            for matched_text, norm_score in normalized_matches.items():
                weighted_score = norm_score * weights[method]
                scores[matched_text] = scores.get(matched_text, 0) + weighted_score

    # Sort the results by the accumulated scores
    ranked_results = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [[text, scores[text]] for text in ranked_results]


print(search_and_rank("search for"))

# https://www.sbert.net/docs/pretrained_models.html

# https://www.sbert.net/examples/applications/cross-encoder/README.html
# https://www.sbert.net/docs/pretrained_cross-encoders.html

# https://www.sbert.net/docs/usage/semantic_textual_similarity.html
# https://www.sbert.net/examples/applications/retrieve_rerank/README.html
# https://www.sbert.net/examples/applications/semantic-search/README.html

# https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0
# roberta-large-nli-stsb-mean-tokens - STSb performance: 86.39
# roberta-base-nli-stsb-mean-tokens - STSb performance: 85.44
# bert-large-nli-stsb-mean-tokens - STSb performance: 85.29
# distilbert-base-nli-stsb-mean-tokens - STSb performance: 85.16

# Trie or inverted indexes

# https://www.datacamp.com/tutorial/stemming-lemmatization-python
# https://www.guru99.com/stemming-lemmatization-python-nltk.html

# https://huggingface.co/BAAI/bge-large-en-v1.5

# https://github.com/ssut/py-googletrans
# https://github.com/suqingdong/googletranslatepy

# Elmo
# InferSent
# Universal sentence encoder multilingual
# Sentence-BERT
# https://huggingface.co/shibing624/text2vec-base-chinese
# https://huggingface.co/GanymedeNil/text2vec-large-chinese
# https://pypi.org/project/transvec/
# https://stackoverflow.com/questions/62385002/latest-pre-trained-multilingual-word-embedding

"""
Discuss how a text search program can use several matching algorithms to retrieve and rank the results.

Certainly! When a text search program needs to retrieve and rank results, it typically employs a combination of different matching algorithms to ensure accuracy and relevance. Here's a brief overview:

1. *Exact String Matching*: This is the most basic form of text search. Algorithms like Boyer-Moore, Knuth-Morris-Pratt, or the Rabin-Karp algorithm can be used to quickly find exact matches of a query string in a text.

2. *Regular Expression Matching*: Allows users to search using patterns. For instance, searching for "a.b" would match "aob", "acb", "adb", etc.

3. *Stemming and Lemmatization*: These are techniques to reduce words to their root form. For instance, "running", "runner", and "ran" might all be reduced to "run". This can help in finding relevant results even if the exact word form doesn't match.

4. **Term Frequency-Inverse Document Frequency (TF-IDF)**: This algorithm weighs words based on their importance. Common words like "and", "the", and "is" are given less weight, while rare words are given more. This helps rank documents by relevance.

5. *Vector Space Models*: This includes algorithms like cosine similarity where documents and query strings are transformed into vectors in a multi-dimensional space. The closeness (cosine of the angle) between the document and query vectors is used to rank results.

6. **Latent Semantic Analysis (LSA)**: This technique uses singular value decomposition on the term-document matrix to identify patterns and relationships between terms and concepts in unstructured data.

7. *Word Embeddings*: Techniques like Word2Vec or GloVe create vector representations of words that capture their meanings and relationships with other words. These vectors can be used to understand the semantic similarity between search queries and documents.

8. *Ranking Algorithms*: Once results are retrieved, they need to be ranked. Algorithms like Google's PageRank consider the importance of each page based on the number and quality of links to it.

9. *Fuzzy Matching*: Algorithms like the Levenshtein distance measure the "distance" between two strings, allowing for minor mismatches. This is useful for catching typos or slightly different wordings.

10. *N-gram Matching*: This divides text into chunks (e.g., bigrams are 2-letter chunks, trigrams are 3, and so on). It can be especially useful for partial matches or when the exact sequence of words is uncertain.

11. *Boolean Operators*: These allow users to refine their search by combining terms (AND, OR) or excluding terms (NOT).

To effectively retrieve and rank results:

- The search program first identifies potentially relevant documents using techniques like exact string matching, regular expression matching, or n-gram matching.
  
- Next, it refines and ranks these results using more sophisticated algorithms like TF-IDF, vector space models, or word embeddings.
  
- Finally, meta-information like user behavior, click-through rates, or external factors like PageRank might be considered to further fine-tune the rankings.

By using a combination of these matching and ranking algorithms, a text search program can deliver relevant and accurate results to users.
"""
