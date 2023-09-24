


from functools import lru_cache

from googletrans import Translator

translator = Translator()

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
    print(f"Generated {len(ngrams)} n-grams")
    print(f"Cache info: {get_all_ngrams.cache_info()}")
    print(ngrams)
    return ngrams


def search(text: str, query: str) -> list[tuple[str, int]]:
    """Search for the query in the text."""
    ngrams = get_all_ngrams(text)
    return [ngram for ngram in ngrams if query in ngram[0]]

def main():
    text = "它 用于 搜索 含义 相似 的 单词 ， 但 不 能 识别 同义词 。"
    query = "搜索"
    print(search(text, query))

if __name__ == '__main__':
    main()