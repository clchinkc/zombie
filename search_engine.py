import string
from collections import defaultdict

import jieba
import nltk
from fuzzywuzzy import process
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
        self.frequency = 0  # To track the frequency of words

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, frequency=1):
        # Validate input
        if not word or len(word) > 100 or not all(char.isalnum() or char.isspace() for char in word):
            raise ValueError("Invalid word. Ensure it's alphanumeric, not too long, and not empty.")
        
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.frequency += frequency
        node.is_end_of_word = True

    def batch_insert(self, words):
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        for word, freq in word_freq.items():
            self.insert(word, freq)

    def suggest_helper(self, node, prefix, suggestions, limit):
        if len(suggestions) >= limit:
            return
        if node.is_end_of_word:
            suggestions.append((prefix, node.frequency))
        for char, next_node in node.children.items():
            self.suggest_helper(next_node, prefix + char, suggestions, limit)

    def autocomplete(self, prefix, limit=10):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  # No suggestions if prefix not in trie

        suggestions = []
        self.suggest_helper(node, prefix, suggestions, limit)

        # Sort by frequency and relevance
        return sorted(suggestions, key=lambda x: -x[1])[:limit]



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
        self.inverted_index = InvertedIndex()
        self.content = []  # Holds searchable content
        self.english_lemmatizer = WordNetLemmatizer()

    def index_corpus(self, corpus, trie):
        trie.batch_insert([word.lower() for word in corpus])

    def detect_language(self, text):
        ascii_chars = sum(1 for char in text if char in string.printable)
        non_ascii_chars = len(text) - ascii_chars
        return 'english' if ascii_chars > non_ascii_chars else 'chinese'

    def get_autocomplete_suggestions(self, prefix):
        language = self.detect_language(prefix)
        trie = self.english_trie if language == 'english' else self.chinese_trie
        return trie.autocomplete(prefix.lower() if language == 'english' else prefix)

    def process_text(self, text):
        language = self.detect_language(text)
        if language == 'english':
            return [self.english_lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(text)]
        else:  # language == 'chinese'
            return [char for char in text]

    def add_content(self, content):
        if not content:
            raise ValueError("Content cannot be empty.")
        
        document_id = len(self.content)
        words = self.process_text(content)
        for word in words:
            self.inverted_index.add(document_id, word)
        self.content.append(content)

    def search_content(self, query):
        if not query:
            return []
        try:
            language = self.detect_language(query)
            query_words = self.process_text(query)
            document_ids = self.perform_search(query_words)
            return [self.content[doc_id] for doc_id in document_ids]
        except Exception as e:
            # Handle any unexpected errors gracefully
            print(f"An error occurred during search: {e}")
            return []

    def perform_search(self, query_words):
        document_ids = set()
        first_query = True
        for word in query_words:
            word_document_ids = self.inverted_index.search(word)
            if first_query:
                document_ids = word_document_ids
                first_query = False
            else:
                document_ids &= word_document_ids  # Intersection
        return document_ids

# Function to extract Chinese words
def get_chinese_words():
    return [''.join(word) for word, _ in sinica_treebank.tagged_words()]

# Load the English and Chinese words from NLTK's corpus
english_words = words.words()
chinese_words = get_chinese_words()

# Initialize the search engine with English and Chinese corpora
search_engine = TextSearchEngine(english_words, chinese_words)

# Add some sample content
contents_to_add = [
    "Hello, welcome to the search engine.",
    "This is a simple example to demonstrate text searching.",
    "Feel free to try searching for words or phrases.",
    "Integration with NLTK's corpus is quite seamless.",
    "歡迎使用搜尋引擎。",
    "這是一個簡單的範例來示範文字搜尋。",
    "隨意嘗試搜尋詞語或短語。",
    "與 NLTK 語料庫的整合非常無縫。"
]

for content in contents_to_add:
    search_engine.add_content(content)

# Testing the autocomplete function
english_prefix = "hel"
chinese_prefix = "我"

english_suggestions = search_engine.get_autocomplete_suggestions(english_prefix)
chinese_suggestions = search_engine.get_autocomplete_suggestions(chinese_prefix)

print(f"English suggestions for prefix '{english_prefix}': {english_suggestions[:10]}")
print(f"Chinese suggestions for prefix '{chinese_prefix}': {chinese_suggestions[:10]}")

# Testing the search functionality
english_query = "search"
chinese_query = "搜"

english_search_results = search_engine.search_content(english_query)
chinese_search_results = search_engine.search_content(chinese_query)

print(f"Search results for English query '{english_query}':")
for result in english_search_results:
    print("-", result)

print(f"Search results for Chinese query '{chinese_query}':")
for result in chinese_search_results:
    print("-", result)

"""
To enhance the functionality and performance of the Text Search Engine, several key improvements can be considered:

1. **Advanced Query Processing and Text Processing**:
    - Implement sophisticated query processing capabilities, like handling synonyms, near-matches, or fuzzy searches, to improve the search experience.
    - For Chinese text, introduce processing steps akin to lemmatization or stemming. This requires handling complex aspects of the language, such as idioms, phrases, and context-aware suggestions.
    - Consider incorporating advanced text processing features, especially for Chinese text, like handling phrases, idioms, and context-aware suggestions.

2. **Efficiency and Memory Optimization**:
    - Optimize Trie construction for memory usage and efficiency, considering the large size of English and Chinese corpora. Techniques like shared node optimization or using compressed tries or DAWGs could be beneficial.
    - For the InvertedIndex, explore persistent data structures like pyrsistent.PMap that share common data but may introduce additional dependencies.

3. **Enhanced Search Functionalities**:
    - Extend the Inverted Index search to support partial matches, phrase searches, and boolean queries (AND, OR, NOT) for more complex searches.
    - Improve phrase searching to support not just individual word searching but also exact phrase matching.
    - Enable the use of boolean operators in queries for more sophisticated search capabilities.

4. **Autocomplete Functionality Enhancements**:
    - Integrate fuzzy matching using algorithms like Levenshtein distance to handle slight misspellings in the autocomplete function.
    - Expand autocomplete to include multi-word predictions by storing common phrases or bigrams/trigrams in the trie and adjusting the suggestion logic.
    - Optimize the autocomplete method in the Trie implementation for efficiency, such as terminating early if a prefix is not found.

5. **User Interface and Integration**:
    - Plan the integration of the search engine with a user interface, considering how users will interact with it and the feedback they will receive.
    - For practical application, integrating this engine with a web interface would be beneficial.

6. **Handling Synonyms and Homonyms**:
    - Implement functionality to handle synonyms and homonyms, which could involve expanding queries with synonyms or using context to differentiate between meanings of homonyms.

7. **Error Handling and User Feedback**:
    - Enhance error handling and user feedback mechanisms. For instance, provide suggestions or nearest matches when a search term isn't found, instead of just returning an empty set.

8. **Advanced Analytics and Reporting**:
    - Develop an analytics module to understand user search patterns, frequent queries, and areas where the search engine might underperform, guiding further improvements and optimizations.

By addressing these areas, the Text Search Engine can become more robust, efficient, and user-friendly, catering to a wider range of search requirements and languages.
"""
