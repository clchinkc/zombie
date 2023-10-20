
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

# Initialize translator
translator = Translator()

# ---------------------------------- SEGMENTATION ---------------------------------- #

def segment_languages(sentence: str):
    """Segment a sentence into different language sections."""
    pattern = re.compile(r"([a-zA-Z\s]+|[\u4e00-\u9FFF]+)")
    return pattern.findall(sentence)


# ---------------------------------- TEXT CLEANING ---------------------------------- #

def clean_english_text(text: str) -> str:
    """Remove non-alphanumeric characters and tokenize English text."""
    text = re.sub(r"[^a-zA-Z0-9\s\n]", "", text)
    return ' '.join(token for token in word_tokenize(text))

def fullwidth_to_halfwidth(s: str) -> str:
    """Convert full-width characters to half-width characters."""
    return ''.join(chr(ord(char) - 0xfee0 if 0xFF01 <= ord(char) <= 0xFF5E else ord(char) if ord(char) != 0x3000 else 32) for char in s)

def clean_chinese_text(text: str) -> str:
    """Tokenize Chinese text and convert full-width characters."""
    return ' '.join(jieba.lcut(fullwidth_to_halfwidth(text), cut_all=False))

def clean_text_based_on_language(text: str, detected_lang: str) -> str:
    """Clean text based on the specific provided language."""
    if detected_lang == 'en':
        return clean_english_text(text)
    elif detected_lang in ['zh-CN', 'zh-TW']:
        return clean_chinese_text(text)
    else:
        return text

def clean_text(sentence: str) -> str:
    """Clean a mixed-language sentence."""
    segments = segment_languages(sentence)
    
    cleaned_segments = []
    for segment in segments:
        # Skip whitespace segments
        if segment.strip() == '':
            continue
        cleaned_segment = clean_text_based_on_language(segment, translator.detect(segment).lang)
        cleaned_segments.append(cleaned_segment.strip())  # Stripping to ensure no leading or trailing spaces
    
    return ' '.join(cleaned_segments)


# ---------------------------------- TEXT PREPROCESSING ---------------------------------- #

def preprocess_chinese_text(text: str) -> str:
    """Remove non-Chinese characters from Chinese text."""
    text = re.sub(r"[^\u4e00-\u9FFF\s\n]", " ", text)
    return ' '.join(text.split())

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

def preprocess_text_based_on_language(text: str, detected_lang: str) -> str:
    """Preprocess text based on the specific provided language."""
    if detected_lang == 'en':
        return preprocess_english_text(text)
    elif detected_lang in ['zh-CN', 'zh-TW']:
        return preprocess_chinese_text(text)
    else:
        return text  # leave other languages unchanged

def preprocess_text(text: str) -> str:
    """Preprocess mixed-language text."""
    segments = segment_languages(text)
    preprocessed_segments = []
    for segment in segments:
        # Skip whitespace segments
        if segment.strip() == '':
            continue
        preprocessed_segment = preprocess_text_based_on_language(segment, translator.detect(segment).lang)
        preprocessed_segments.append(preprocessed_segment.strip())
    return ' '.join(preprocessed_segments)


# ---------------------------------- LANGUAGE TRANSLATION ---------------------------------- #

def translate_segment(segment: str, target_lang: str):
    """Translate a segment to the desired target language."""
    detected_lang = translator.detect(segment).lang
    if detected_lang != target_lang:
        return translator.translate(segment, dest=target_lang).text
    return segment

def translate_multilanguage(sentence: str):
    """Translate mixed-language sentences."""
    segments = segment_languages(sentence)
    english_translation = ' '.join([translate_segment(segment, 'en').strip() for segment in segments])
    traditional_translation = ' '.join([translate_segment(segment, 'zh-TW').strip() for segment in segments])
    simplified_translation = ' '.join([translate_segment(segment, 'zh-CN').strip() for segment in segments])
    
    return sentence, english_translation, traditional_translation, simplified_translation


original_keyword = "搜尋 for"

# Translation using the function
original_translated, english_translated, traditional_translated, simplified_translated = translate_multilanguage(original_keyword)

print("Original keyword:", original_translated)
print("English Translation:", english_translated)
print("Traditional Chinese Translation:", traditional_translated)
print("Simplified Chinese Translation:", simplified_translated)

cleaned_original = clean_text(original_translated)
print("\nCleaned Original keyword:", cleaned_original)

cleaned_english = clean_text(english_translated)
print("Cleaned English keyword:", cleaned_english)
preprocessed_english = preprocess_text(cleaned_english)
print("Preprocessed English keyword:", preprocessed_english)

cleaned_traditional = clean_text(traditional_translated)
print("Cleaned Traditional keyword:", cleaned_traditional)
preprocessed_traditional = preprocess_text(cleaned_traditional)
print("Preprocessed Traditional keyword:", preprocessed_traditional)

cleaned_simplified = clean_text(simplified_translated)
print("Cleaned Simplified keyword:", cleaned_simplified)
preprocessed_simplified = preprocess_text(cleaned_simplified)
print("Preprocessed Simplified keyword:", preprocessed_simplified)






