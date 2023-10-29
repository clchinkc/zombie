import nltk
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Initial setup for NLP tools
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('vader_lexicon')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()



def scrape_news(ticker):
    """Scrape Yahoo Finance news headlines for a given ticker."""
    URL = f'https://finance.yahoo.com/quote/{ticker}/news?p={ticker}'
    HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    page = requests.get(URL, headers=HEADERS)
    soup = BeautifulSoup(page.content, 'html.parser')

    headlines = []
    for item in soup.find_all('h3', class_='Mb(5px)'):
        a_tag = item.find('a')
        if a_tag and a_tag.text:
            headlines.append(a_tag.text)
    
    return headlines


def process_text(text):
    """Process text for sentiment analysis."""
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

def get_sentiment(text):
    """Determine sentiment of processed text."""
    sentiment = sia.polarity_scores(text)
    return sentiment

if __name__ == "__main__":
    FAAMG = ['FB', 'AAPL', 'AMZN', 'MSFT', 'GOOGL']

    for ticker in FAAMG:
        print(f"--- {ticker} ---")
        headlines = scrape_news(ticker)

        for headline in headlines:
            processed_headline = process_text(headline)
            sentiment = get_sentiment(processed_headline)
            print(f"[{sentiment['compound']}] {headline}")

"""
Stock Market Analysis:

Compare the sentiment scores with stock price movements for each company. For instance, a high average positive sentiment score might correlate with an upward movement in stock price.
Build a model using historical sentiment scores and stock prices to predict future stock prices.
Alert System:

Set a threshold for sentiment scores. Whenever the score goes beyond this threshold (either too positive or too negative), send an alert. This could be useful for traders to pay attention to certain news headlines.
Visualizations:

Plot sentiment scores over time to visually understand the sentiment trend for each company.
Create pie charts showing the percentage of positive, negative, and neutral headlines for each company over a given period.
Expand Scope:

Extend the code to cover more companies, sectors, or news sources. This would provide a broader view of the market sentiment.
Instead of just headlines, you can scrape the full articles and analyze their sentiment. This would provide a more in-depth sentiment analysis.
Refined Text Analysis:

Apply topic modeling (e.g., using LDA) to understand the common topics in the news and then analyze sentiment on a topic basis.
Consider named entity recognition (NER) to identify mentions of other companies, people, or products in the headlines, which could be crucial for sentiment interpretation.
Integration with Portfolio Management:

If you have a stock portfolio, integrate the sentiment scores to influence your buy/sell decisions. For instance, consistently negative news sentiment might be a sign to reconsider holding a particular stock.
Historical Analysis:

Store the scraped headlines and sentiment scores in a database. Over time, you can analyze historical sentiment trends and correlate them with historical stock price movements.
Comparison with Other Sentiment Analysis Tools:

There are many sentiment analysis tools and libraries available. By comparing the results from different tools, you can achieve a more robust understanding of the true sentiment.
Automate & Schedule:

Automate the script to run at specific intervals (e.g., daily or weekly) to keep track of sentiment over time.
Feedback Loop:

If you're using this for trading, keep track of decisions made based on the sentiment scores. Over time, refine your thresholds or models based on actual outcomes to improve prediction accuracy.
"""