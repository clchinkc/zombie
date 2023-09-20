import re

import jieba
import nltk
from googletrans import Translator
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# You can uncomment the next lines if you haven't downloaded the necessary resources yet
# nltk.download('punkt')
# nltk.download('wordnet')

translator = Translator()

def preprocess_english_text(text):
    # Remove all non-alphanumeric characters except for spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Convert all text to lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Stemming
    stemmer = SnowballStemmer("english")
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(stemmed_token) for stemmed_token in stemmed_tokens]
    # Join the tokens back into a single string for search
    return ' '.join(lemmatized_tokens)


def fullwidth_to_halfwidth(s):
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        n.append(chr(num))
    return ''.join(n)


def preprocess_chinese_text(text):
    text = fullwidth_to_halfwidth(text)
    tokens = list(jieba.cut(text))
    return ' '.join(tokens)


def preprocess_text(text):
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