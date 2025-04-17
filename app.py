import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os
from datetime import datetime
import time
from dotenv import load_dotenv

load_dotenv()

# Represents a news article with metadata and fields for summary and sentiment
class NewsArticle:
    def __init__(self, title, url, content, source, published_date):
        self.title = title
        self.url = url
        self.content = content
        self.source = source
        self.published_date = published_date
        self.summary = None
        self.sentiment = None

# Scrapes news articles using the NewsAPI and fetches article content
class NewsAPIScraper:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("NEWS_API_KEY")
        if not self.api_key:
            raise ValueError("NEWS_API_KEY is required")
        self.base_url = "https://newsapi.org/v2/top-headlines"
        
    def fetch_news(self, category='technology', country='us', page_size=5):
        params = {
            'category': category,
            'country': country,
            'pageSize': page_size,
            'apiKey': self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        if response.status_code != 200:
            print(f"Error fetching news: {response.status_code}")
            return []
            
        data = response.json()
        articles = []
        
        for article in data['articles']:
            content = self.fetch_article_content(article['url']) if article['url'] else ""
            if not content and article.get('description'):
                content = article['description']
                
            news_article = NewsArticle(
                title=article['title'],
                url=article['url'],
                content=content,
                source=article['source']['name'] if article['source'] else "Unknown",
                published_date=article['publishedAt']
            )
            articles.append(news_article)
            
        return articles
        
    def fetch_article_content(self, url):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = soup.find_all('p')
                content = ' '.join([p.text for p in paragraphs])
                return content
            return ""
        except Exception as e:
            print(f"Error fetching article content: {e}")
            return ""

# Loads and manages a transformer-based model to generate text summaries
class SummarizerModel:
    def __init__(self, model_name="t5-small"):
        print(f"Loading {model_name} model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)
        self.max_token_length = self.tokenizer.model_max_length
        print("Model loaded successfully")
        
    def summarize(self, text, max_length=150, min_length=30):
        if not text or len(text) < min_length:
            return "Text too short to summarize."
            
        chunks = self._chunk_text(text)
        summaries = []
        
        for chunk in chunks:
            # Check token length and truncate if necessary
            encoded = self.tokenizer(chunk, truncation=True, max_length=self.max_token_length-2, return_tensors="pt")
            chunk_text = self.tokenizer.decode(encoded['input_ids'][0], skip_special_tokens=True)
            
            chunk_max_length = max(30, max_length//len(chunks))
            chunk_min_length = max(10, min_length//len(chunks))
            
            summary = self.summarizer(chunk_text, max_length=chunk_max_length, min_length=chunk_min_length)
            summaries.append(summary[0]['summary_text'])
            
        return " ".join(summaries)
    
    def _chunk_text(self, text, max_words=300):
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_words):
            chunk = ' '.join(words[i:i + max_words])
            chunks.append(chunk)
            
        return chunks

# Analyzes sentiment of text using transformer-based sentiment analysis models
class SentimentAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.max_token_length = 512  # Default for most sentiment models
        
    def analyze(self, text):
        if not text:
            return {"label": "NEUTRAL", "score": 0.5}
            
        chunks = self._chunk_text(text)
        results = []
        
        for chunk in chunks:
            # Ensure chunk doesn't exceed token limit
            words = chunk.split()
            if len(words) > 100:  # Approximate token count
                chunk = ' '.join(words[:100])
                
            result = self.sentiment_analyzer(chunk)[0]
            results.append(result)
            
        # Aggregate sentiment across chunks
        positive_count = sum(1 for r in results if r['label'] == 'POSITIVE')
        negative_count = sum(1 for r in results if r['label'] == 'NEGATIVE')
        neutral_count = len(results) - positive_count - negative_count
        
        # Determine dominant sentiment
        if positive_count > negative_count and positive_count > neutral_count:
            avg_score = sum(r['score'] for r in results if r['label'] == 'POSITIVE') / max(1, positive_count)
            return {"label": "POSITIVE", "score": avg_score}
        elif negative_count > positive_count and negative_count > neutral_count:
            avg_score = sum(r['score'] for r in results if r['label'] == 'NEGATIVE') / max(1, negative_count)
            return {"label": "NEGATIVE", "score": avg_score}
        else:
            return {"label": "NEUTRAL", "score": 0.5}
    
    def _chunk_text(self, text, max_words=100):
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_words):
            chunk = ' '.join(words[i:i + max_words])
            chunks.append(chunk)
            
        return chunks

# Orchestrates the entire news summarization workflow
class NewsSummarizer:
    def __init__(self):
        self.news_scraper = NewsAPIScraper()
        self.summarizer = SummarizerModel()
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def get_news_summaries(self, category='technology', country='us', page_size=5):
        articles = self.news_scraper.fetch_news(category, country, page_size)
        
        for article in articles:
            if article.content:
                article.summary = self.summarizer.summarize(article.content)
                article.sentiment = self.sentiment_analyzer.analyze(article.content)
                
        return articles
        
    def display_results(self, articles):
        print("\n" + "="*80)
        print(f"NEWS SUMMARIES ({datetime.now().strftime('%Y-%m-%d %H:%M')})")
        print("="*80)
        
        for i, article in enumerate(articles, 1):
            print(f"\n{i}. {article.title}")
            print(f"Source: {article.source} | {article.published_date}")
            print(f"URL: {article.url}")
            print("-" * 40)
            
            if article.summary:
                print("SUMMARY:")
                print(article.summary)
            else:
                print("No summary available.")
                
            if article.sentiment:
                sentiment_emoji = "😊" if article.sentiment['label'] == 'POSITIVE' else "😐" if article.sentiment['label'] == 'NEUTRAL' else "😟"
                print(f"\nSENTIMENT: {article.sentiment['label']} {sentiment_emoji} (score: {article.sentiment['score']:.2f})")
                
            print("-" * 80)
    
    def save_to_csv(self, articles, filename=None):
        if not articles:
            print("No articles to save.")
            return
            
        if not filename:
            date_str = datetime.now().strftime('%Y%m%d_%H%M')
            filename = f"news_summaries_{date_str}.csv"
            
        data = []
        for article in articles:
            data.append({
                'Title': article.title,
                'Source': article.source,
                'Date': article.published_date,
                'URL': article.url,
                'Summary': article.summary,
                'Sentiment': article.sentiment['label'] if article.sentiment else 'N/A',
                'Sentiment Score': article.sentiment['score'] if article.sentiment else 'N/A'
            })
            
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Saved {len(articles)} summaries to {filename}")


def main():
    print("AI News Summarizer Bot")
    print("======================")
    
    try:
        news_bot = NewsSummarizer()
        
        category = input("Enter news category (technology, business, health, science, entertainment, sports): ").lower() or "technology"
        country = input("Enter country code (us, gb, in, au, ca): ").lower() or "us"
        page_size = int(input("Number of articles to fetch (1-10): ") or "5")
        
        valid_categories = ["technology", "business", "health", "science", "entertainment", "sports"]
        valid_countries = ["us", "gb", "in", "au", "ca"]
        
        if category not in valid_categories:
            print(f"Invalid category. Using default: technology")
            category = "technology"
            
        if country not in valid_countries:
            print(f"Invalid country code. Using default: us")
            country = "us"
            
        if not 1 <= page_size <= 10:
            print("Invalid number of articles. Using default: 5")
            page_size = 5
        
        print(f"\nFetching {page_size} {category} news articles from {country.upper()}...")
        start_time = time.time()
        
        articles = news_bot.get_news_summaries(category, country, page_size)
        
        elapsed_time = time.time() - start_time
        print(f"Processed {len(articles)} articles in {elapsed_time:.2f} seconds")
        
        news_bot.display_results(articles)
        
        save_option = input("\nSave results to CSV? (y/n): ").lower()
        if save_option == 'y':
            news_bot.save_to_csv(articles)
            
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()