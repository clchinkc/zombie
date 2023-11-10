from collections import defaultdict

import jieba
import nltk
from nltk.corpus import sinica_treebank, words
from nltk.stem import WordNetLemmatizer

# Ensure the necessary NLTK corpora are downloaded
nltk.download('words', quiet=True)
nltk.download('sinica_treebank', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Or wordfreq
# https://www.hankcs.com/program/python/nltk-chinese-corpus-sinica_treebank.html

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def suggest_helper(self, node, prefix, suggestions):
        if node.is_end_of_word:
            suggestions.append(prefix)
        for char, next_node in node.children.items():
            self.suggest_helper(next_node, prefix + char, suggestions)

    def autocomplete(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  # No autocomplete suggestions if prefix not in trie
        suggestions = []
        self.suggest_helper(node, prefix, suggestions)
        return suggestions

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
    
    def add(self, document_id, word):
        self.index[word].add(document_id)
    
    def search(self, query):
        return self.index.get(query, set())

class TextSearchEngine:
    def __init__(self, english_corpus, chinese_corpus):
        self.english_trie = Trie()
        self.chinese_trie = Trie()
        self.index_corpus(english_corpus, self.english_trie)
        self.index_corpus(chinese_corpus, self.chinese_trie)
        self.english_inverted_index = InvertedIndex()
        self.chinese_inverted_index = InvertedIndex()
        self.english_content = []  # This will hold searchable English content
        self.chinese_content = []  # This will hold searchable Chinese content
        self.english_lemmatizer = WordNetLemmatizer()

    def index_corpus(self, corpus, trie):
        for word in corpus:
            trie.insert(word.lower())

    def get_autocomplete_suggestions(self, prefix, language='english'):
        if language == 'english':
            return self.english_trie.autocomplete(prefix.lower())
        elif language == 'chinese':
            return self.chinese_trie.autocomplete(prefix)
        else:
            raise ValueError("Unsupported language.")

    def index_document(self, document_id, text, inverted_index, language='english'):
        if language == 'english':
            words = text.lower().split()
            words = [self.english_lemmatizer.lemmatize(word) for word in words]
        elif language == 'chinese':
            words = [char for char in text]
        else:
            raise ValueError("Unsupported language.")
        
        for word in words:
            inverted_index.add(document_id, word)

    def add_content(self, content, language='english'):
        document_id = len(self.english_content) if language == 'english' else len(self.chinese_content)
        
        if language == 'english':
            self.english_content.append(content)
            self.index_document(document_id, content, self.english_inverted_index, language)
        elif language == 'chinese':
            self.chinese_content.append(content)
            self.index_document(document_id, content, self.chinese_inverted_index, language)
        else:
            raise ValueError("Unsupported language.")

    def search_content(self, query, language='english'):
        results = []
        if language == 'english':
            query_lemmatized = self.english_lemmatizer.lemmatize(query.lower())
            document_ids = self.english_inverted_index.search(query_lemmatized)
            results = [self.english_content[doc_id] for doc_id in document_ids]
        elif language == 'chinese':
            document_ids = self.chinese_inverted_index.search(query)
            results = [self.chinese_content[doc_id] for doc_id in document_ids]
        else:
            raise ValueError("Unsupported language.")
        return results




# Function to extract Chinese words
def get_chinese_words():
    return [''.join(word) for word, _ in sinica_treebank.tagged_words()]

# Load the English and Chinese words from NLTK's corpus
english_words = words.words()
chinese_words = get_chinese_words()

# Initialize the search engine with English and Chinese corpora
search_engine = TextSearchEngine(english_words, chinese_words)

# Add some sample English and Chinese content
search_engine.add_content("Hello, welcome to the search engine.", 'english')
search_engine.add_content("This is a simple example to demonstrate text searching.", 'english')
search_engine.add_content("Feel free to try searching for words or phrases.", 'english')
search_engine.add_content("Integration with NLTK's corpus is quite seamless.", 'english') # Not related

search_engine.add_content("歡迎使用搜尋引擎。", 'chinese')
search_engine.add_content("這是一個簡單的範例來示範文字搜尋。", 'chinese')
search_engine.add_content("隨意嘗試搜尋詞語或短語。", 'chinese')
search_engine.add_content("與 NLTK 語料庫的整合非常無縫。", 'chinese') # Not related

# Testing the autocomplete function
english_prefix = "hel"
chinese_prefix = "我"

english_suggestions = search_engine.get_autocomplete_suggestions(english_prefix, 'english')
chinese_suggestions = search_engine.get_autocomplete_suggestions(chinese_prefix, 'chinese')

print(f"English suggestions for prefix '{english_prefix}': {english_suggestions[:10]}")  # Displaying only top 10
print(f"Chinese suggestions for prefix '{chinese_prefix}': {chinese_suggestions[:10]}")  # Displaying only top 10

# Testing the search functionality
english_query = "search"
chinese_query = "搜"

english_search_results = search_engine.search_content(english_query, 'english')
chinese_search_results = search_engine.search_content(chinese_query, 'chinese')

print(f"Search results for English query '{english_query}':")
for result in english_search_results:
    print("-", result)

print(f"Search results for Chinese query '{chinese_query}':")
for result in chinese_search_results:
    print("-", result)