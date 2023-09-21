

import re

import jieba
import nltk
from googletrans import Translator
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# You can uncomment the next lines if you haven't downloaded the necessary resources yet
# nltk.download('punkt')
# nltk.download('wordnet')

translator = Translator()

def fullwidth_to_halfwidth(s: str) -> str:
    """Convert full-width characters to half-width characters."""
    return ''.join(chr(ord(char) - 0xfee0 if 0xFF01 <= ord(char) <= 0xFF5E else ord(char) if ord(char) != 0x3000 else 32) for char in s)

def preprocess_chinese_text(text: str) -> str:
    """Tokenize Chinese text."""
    return ' '.join(list(jieba.cut(fullwidth_to_halfwidth(text))))

def preprocess_english_text(text: str) -> str:
    """Preprocess English text - removal of non-alphanumeric characters, conversion to lowercase, tokenization, and lemmatization."""
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text).lower()
    lemmatizer = WordNetLemmatizer()
    return ' '.join(lemmatizer.lemmatize(token) for token in word_tokenize(text))


def preprocess_text(text):
    """Detect the language of the text and preprocess accordingly."""
    detected_lang = translator.detect(text).lang
    if detected_lang == 'zh-CN' or detected_lang == 'zh-TW':
    # Simple heuristic: If the text contains any Chinese characters, use the Chinese preprocessing
    # if re.search("[\u4e00-\u9FFF]", text):
        return preprocess_chinese_text(text)
    elif detected_lang == 'en':
        return preprocess_english_text(text)
    else:
        return text

# Test the function
sample_text_en = "Cats are running faster than dogs!"
result_en = preprocess_text(sample_text_en)
print(result_en)

sample_text_zh = "我愛吃蘋果和香蕉。"
result_zh = preprocess_text(sample_text_zh)
print(result_zh)





