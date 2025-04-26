# 🌟 AI News Summarizer Bot

The **AI News Summarizer Bot** is a Python-based application that fetches the latest news articles, summarizes their content using transformer-based models, and analyzes their sentiment. It provides a streamlined way to stay updated with news while saving time by summarizing lengthy articles.

---

## ✨ Features

- 📰 **News Scraping**: Fetches news articles from the NewsAPI.
- ✍️ **Content Summarization**: Summarizes articles using a transformer-based model (e.g., T5).
- 📊 **Sentiment Analysis**: Analyzes the sentiment of articles (Positive, Neutral, or Negative).
- 📂 **CSV Export**: Saves summarized articles and sentiment analysis results to a CSV file.
- ⚙️ **Customizable**: Allows users to specify news categories, countries, and the number of articles to fetch.

---

## 🛠️ Requirements

- **Python**: Version 3.8 or higher
- Install dependencies using the following command:

```bash
pip install -r requirements.txt
```

---

## 📂 Project Structure

```
.
├── app.py               # Main application script
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (not included in version control)
├── .gitignore           # Git ignore file
├── LICENSE              # License file
└── README.md            # Project documentation
```

---

## 🌍 Environment Setup

Create a `.env` file in the root directory and add the following:

```env
NEWS_API_KEY=your_news_api_key_here
```

Replace `your_news_api_key_here` with your API key from [NewsAPI](https://newsapi.org/).

---

## 🚀 Usage

Run the application with:

```bash
python app.py
```

### Interactive Workflow:
1. Select a news category (e.g., technology, business, health).
2. Choose a country (e.g., US, GB, IN).
3. Specify the number of articles to fetch (1–10).
4. View summarized articles and sentiment analysis.
5. Optionally save results to a CSV file.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙏 Acknowledgments

- [NewsAPI](https://newsapi.org/) for providing the news data.
- [Hugging Face Transformers](https://huggingface.co/transformers/) for the summarization and sentiment analysis models.