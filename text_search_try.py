
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

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

# Sample text for demonstration
sample_text = """
Fuzzy searches help in finding words that are close in spelling. Semantic searches is more complex.
It is used to search for words that are similar in meaning.
We are trying to find exact and inexact matches.
This is the end of the sample text.
"""

# Initialize translator
translator = Translator()

# setup Chroma in-memory, for easy prototyping. Can add persistence easily
# client = chromadb.Client()
# client = chromadb.EphemeralClient()
# setup Chroma with persistence, for production use. Can also use a remote database
client = chromadb.PersistentClient(path="chroma.db", settings=Settings(allow_reset=True))

client.reset() # reset the database

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="distiluse-base-multilingual-cased-v1")


# Available models for SentenceTransformer('model_name'), return model.encode(sentence)
# flax-sentence-embeddings/all_datasets_v3_mpnet-base (update from q&a platform)
# bert-base-nli-mean-tokens (original SBERT)
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

vectorizer = TfidfVectorizer(analyzer='word')

tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
model = BertModel.from_pretrained('google/bert_uncased_L-2_H-128_A-2')
# bert-base-uncased
# bert-base-chinese
# bert-base-multilingual-cased
# bert-base-cased
# ckiplab/bert-base-chinese-ner

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
    lang = translator.detect(text).lang
    words = text.split()
    if lang == 'en':
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
    elif lang == 'zh-TW' or lang == 'zh-CN':
        return [''.join(words[i:i+n]) for i in range(len(words) - n + 1)]
    else:
        raise ValueError(f"Unsupported language: {lang}")

@lru_cache(maxsize=100)
def get_all_ngrams(text: str) -> list[tuple[str, int]]:
    """Generate all n-grams from the text."""
    words = text.split()
    num_words = len(words)
    ngrams = [(gram, n) for n in range(1, num_words + 1) for gram in get_ngrams(text, n)]
    return ngrams

@lru_cache(maxsize=100)
def get_word_embeddings(sentence: str) -> torch.Tensor:
    """Get BERT embedding for a given sentence."""
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs['last_hidden_state'][:, 0, :]

@lru_cache(maxsize=100)
def word_embedding_similarity(word1_embedding: torch.Tensor, word2_embedding: torch.Tensor) -> float:
    similarity = torch.nn.functional.cosine_similarity(word1_embedding, word2_embedding)
    return similarity.item()




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



def fullwidth_to_halfwidth(s: str) -> str:
    """Convert full-width characters to half-width characters."""
    return ''.join(chr(ord(char) - 0xfee0 if 0xFF01 <= ord(char) <= 0xFF5E else ord(char) if ord(char) != 0x3000 else 32) for char in s)

def clean_chinese_text(text: str) -> str:
    """Tokenize Chinese text."""
    return ' '.join(jieba.lcut(fullwidth_to_halfwidth(text), cut_all=False))

def clean_english_text(text: str) -> str:
    """Remove non-alphanumeric characters and tokenize."""
    text = re.sub(r"[^a-zA-Z0-9\s\n]", "", text)
    return ' '.join(token for token in word_tokenize(text))

def clean_text(text):
    """Detect the language of the text and clean accordingly."""
    detected_lang = translator.detect(text).lang
    if detected_lang in ['zh-CN', 'zh-TW']:
    # Simple heuristic: If the text contains any Chinese characters, use the Chinese cleaning
    # if re.search("[\u4e00-\u9FFF]", text):
        return clean_chinese_text(text)
    elif detected_lang == 'en':
        return clean_english_text(text)
    else:
        return text

def preprocess_chinese_text(text: str) -> str:
    """Remove non-Chinese characters from Chinese text."""
    text = re.sub(r"[^\u4e00-\u9FFF\s\n]", " ", text)
    return text

def get_wordnet_pos(treebank_tag):
    """Map treebank POS tag to first character used by WordNetLemmatizer."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

def preprocess_english_text(text: str) -> str:
    """Lemmatize English text and convert to lowercase."""
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in pos_tags]
    return ' '.join(lemmatized_tokens)

def preprocess_text(text):
    """Detect the language of the text and refine accordingly."""
    detected_lang = translator.detect(text).lang
    if detected_lang == 'en':
        return preprocess_english_text(text)
    elif detected_lang in ['zh-CN', 'zh-TW']:
        return preprocess_chinese_text(text)
    else:
        return text

def translate_multilanguage(sentence):
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



def search_and_rank(keyword, text=sample_text, preprocess=False, weights={'exact': 0.9, 'ngram': 0.6, 'regex': 0.7, 'fuzzy': 0.8, 'tfidf': 0.7, 'bm25': 0.7, 'word_embedding': 0.8, 'semantic': 1.0}):

    # Split the text once
    lines = [line for line in text.split("\n") if line.strip()]
    
    # Get translations of the keyword using translate_multilanguage
    english_keyword, traditional_keyword, simplified_keyword = translate_multilanguage(keyword)
    
    translations = {
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
    else:
        preprocessed_lines = cleaned_lines
        preprocessed_keywords = cleaned_translations

    for lang, kw in preprocessed_keywords.items():
        print(f"Preprocessed {lang.capitalize()} keyword:", kw)
    
    # Create a mapping between processed and original text
    cleaned_to_original_mapping = dict(zip(preprocessed_lines, lines))

    scores = {}

    for lang, kw in preprocessed_keywords.items():
        search_methods = {
            'exact': exact_search(kw, "\n".join(preprocessed_lines)),
            'ngram': ngram_search(kw, "\n".join(preprocessed_lines)),
            'regex': regex_search(kw, "\n".join(preprocessed_lines)),
            'fuzzy': fuzzy_search(kw, "\n".join(preprocessed_lines)),
            'tfidf': tfidf_search(kw, "\n".join(preprocessed_lines)),
            'bm25': bm25_search(kw, "\n".join(preprocessed_lines)),
            'word_embedding': word_embedding_search(kw, "\n".join(preprocessed_lines)),
            'semantic': semantic_search(kw, "\n".join(preprocessed_lines)),
        }

        for method, matches in search_methods.items():
            if not matches:
                continue

            for matched_text, score in matches:
                # Map the matched text back to the original text
                matching_line_original = cleaned_to_original_mapping.get(matched_text, matched_text)

                weighted_score = score * weights[method]
                scores[matching_line_original] = scores.get(matching_line_original, 0) + weighted_score

    # Sort results by the accumulated scores
    ranked_results = sorted(scores, key=lambda x: scores[x], reverse=True)

    return list(zip(ranked_results, [scores[x] for x in ranked_results]))



default_weights = {
    'exact': 0.9,          # Full weight for precise matches.
    'ngram': 0.6,          # Useful for broader matches.
    'regex': 0.7,          # Flexible search with moderate precision.
    'fuzzy': 0.8,          # Useful for variations in spellings.
    'tfidf': 0.7,          # Weighs importance of words in a dataset.
    'bm25': 0.7,           # Weighs importance of words in a dataset.
    'word_embedding': 0.8, # Finds semantically similar terms.
    'semantic': 1.0        # High weight for context and meaning.
}


results = search_and_rank("searches for", sample_text, preprocess=True, weights=default_weights)

print("Results:")
for line, score in results:
    print(f"Line: {line:<{max(len(line) for line, _ in results)}} | Score: {score:>7.3f}")

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

# https://github.com/Friedrich94326/IR_using_chunking

# https://course.spacy.io/zh/

# https://huggingface.co/BAAI/bge-large-en-v1.5

# https://github.com/ssut/py-googletrans
# https://github.com/suqingdong/googletranslatepy

# https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py

# Elmo
# InferSent
# Universal sentence encoder multilingual
# Sentence-BERT
# https://huggingface.co/shibing624/text2vec-base-chinese
# https://huggingface.co/GanymedeNil/text2vec-large-chinese
# https://pypi.org/project/transvec/
# https://stackoverflow.com/questions/62385002/latest-pre-trained-multilingual-word-embedding

# https://blog.csdn.net/FontThrone/article/details/72782499
# https://github.com/fxsjy/jieba
# https://github.com/fxsjy/jieba/issues/7
# https://github.com/ckiplab/ckiptagger
# LLM 分词

# https://levelup.gitconnected.com/building-a-full-text-search-app-using-django-docker-and-elasticsearch-d1bc18504ca4

# https://developers.google.com/cloud-search/docs/guides/improve-search-quality?hl=zh-tw

# https://www.pinecone.io/learn/series/nlp/sentence-embeddings/

# http://norvig.com/spell-correct.html

# Dense Passage Retrievers
# https://www.searchenginejournal.com/generative-retrieval-for-conversational-question-answering/496373/

# https://arxiv.org/pdf/1511.08198.pdf
# https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf
# https://arxiv.org/abs/1502.03520v4
# https://aclanthology.org/W18-3012.pdf
# https://openreview.net/pdf?id=SyK00v5xx
# https://blogs.nlmatics.com/nlp/sentence-embeddings/2020/08/07/Smooth-Inverse-Frequency-Frequency-(SIF)-Embeddings-in-Golang.html
# https://blog.dataiku.com/how-deep-does-your-sentence-embedding-model-need-to-be

# https://yxkemiya.github.io/2016/06/05/coursera-TextRetrievalAndSearchEngines-week1/#more
# https://yxkemiya.github.io/2016/06/08/coursera-TextRetrievalAndSearchEngines-week2-implement-TR/#more
# https://yxkemiya.github.io/2016/06/20/coursera-TextRetrievalAndSearchEngines-week3/#more

# Query Likelihood Retrieval Model
# Divergence-from-randomness model
# PL2 retrieval model
# Boolean vs Vector vs Probabilistic vs Language Models

# siamese network text similarity
# https://zhuanlan.zhihu.com/p/75366208

# https://medium.com/tech-that-works/maximal-marginal-relevance-to-rerank-results-in-unsupervised-keyphrase-extraction-22d95015c7c5
# https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf

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


"""
**Title:** Text Search Program

**Overview:**  
The Text Search Program is an advanced and versatile tool designed to empower users in efficiently locating specific text patterns within a document or dataset. It stands as a comprehensive software solution, integrating sophisticated search techniques, user-friendly interfaces, and customization options to maximize accuracy and productivity in text-based searches.

**Key Features:**

1. **Search Algorithms**: Harnessing cutting-edge search techniques, the program offers:
   - **Exact Match**: Provides results that match the search query perfectly, ensuring precise data retrieval.
   - **Regex (Regular Expression) Search**: Allows users to employ complex patterns, using symbols to match diverse string types.
   - **Fuzzy Search**: Locates approximate matches, accommodating minor variations or typos in the search input.

2. **Customizable Options**: Users can tailor their search process by modifying parameters such as case sensitivity, search depth, and search scope.

3. **Spell Check Integration**: Integrated with a spell-check module, this feature proposes corrections for possible typing errors, ensuring users find the desired results even with misspelled input.

4. **Autocomplete Suggestions**: The program predicts and suggests potential search terms in real-time, based on previous searches and a predefined dictionary, enhancing search speed and accuracy.

5. **Search Result Preview**: Search outcomes are showcased with relevant excerpts, providing users with context to assess the relevance quickly.

6. **User-Friendly Interface**: With an intuitive design, users can easily set their preferences and begin their search journey with minimal hassle.

7. **Search History and Bookmarks**: Users can revisit past searches through a maintained history and can bookmark significant results for easier future access.

8. **Document Handling and Efficiency**: Catering to a variety of formats such as plain text, PDFs, and Word documents, the program ensures swift results even for expansive volumes of text.

9. **Export and Sharing Features**: Search outcomes, accompanied by their context, can be exported for later review or collaborative purposes.

10. **Extensibility and Enhancements**: The program's architecture allows developers to integrate new search algorithms, plugins, and features as required.

**Technical Insights:**  
- To deliver rapid outcomes, efficient string matching algorithms are in place, making it feasible to search large text volumes swiftly.
- The program relies on Python's re module or equivalent for implementing regular expressions.
- Fuzzy searches leverage algorithms such as Levenshtein distance for optimal performance.
- The autocomplete function utilizes a trie data structure or a suitable database for prompt input predictions.

**Additional Features:**
1. **Context Highlighting**: For enhanced clarity, matching text segments within results are highlighted.
2. **Advanced Filters**: Users can refine results by factors like date, text origin, and other pertinent criteria.
3. **Expandable Library**: Supports extensions like additional search algorithms or plugins for boosted capabilities.

**Conclusion:**  
The Text Search Program embodies the pinnacle of text-search solutions, catering to a broad spectrum of searching requirements, be it exact matches or approximate ones. Crafted with a user-centric mindset, it assures that individuals, irrespective of their searching needs, can locate their desired information seamlessly. Whether used for content exploration, data analytics, or academic research, this program stands as a cornerstone tool for any text-based endeavor.
"""

"""
Faceted Search/Navigation. This is the advanced search/filter functionality available on many sites. It's a design pattern. Can read about it here : http://alistapart.com/article/design-patterns-faceted-navigation . Implement it on the back end and the front end if you're bored.

Image Search - FreeCodeCamp calls this Image Search Abstraction Layer which sounds complicated. Instead of interfacing with a 3rd party, make it search a defined path on the local file system. FCC's description : https://www.freecodecamp.com/challenges/image-search-abstraction-layer
"""

"""
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
"""
