# Project Proposal – Option 2: Modeling Experiment

## Title  
**NLP-Powered Financial News Analysis for Market Insights**

## Business Context  
The prices of stocks listed under global exchanges are influenced by a range of factors, with financial performance, innovation, partnerships, and market sentiment playing key roles. In the fast-paced financial industry, news and media coverage can significantly impact investor perception and, in turn, stock price movement. Given the overwhelming volume of news articles and social commentary, investment firms face increasing challenges in staying informed and responding rapidly.

## Objective  

The goal of this project is to build an NLP system that:
  Analyzes the sentiment of financial news articles.
  Summarizes key positive and negative events from weekly news.
  Explores the correlation between news sentiment and stock market trends.
  This system aims to support financial analysts by delivering timely insights into how news impacts stock prices, without promising predictive capabilities beyond what’s implemented.

## Dataset  
https://github.com/ArtZaragozaGitHub/NLP--P6_Sentiment_Analysis_and_Summarization_of_Stock_News/blob/main/stock_news.csv
https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction/data?select=stock_yfinance_data.csv

This dataset includes both unstructured (text) and structured (numerical stock) data.

## Modeling Approach  
The project will be developed in a series of stages:

1. **Data Preprocessing and Exploration**  
   - Clean and normalize financial news text  
   - Align stock prices and news by date  
   - Perform exploratory analysis on sentiment and price patterns

2. **Sentiment Classification**  
    - Text Encoding: Apply various encoding techniques (e.g., TF-IDF or embeddings) to represent news text.
      - Build or fine-tune a sentiment classifier using methods such as:
          - TF-IDF + Logistic Regression (to be decided)
          - sentence transformer embeddings + neural classifiers
    - Evaluate classifier performance using accuracy, F1, precision, and recall

3. **Weekly Sentiment Aggregation**  
     - Aggregate sentiment scores at a weekly level to smooth volatility  
     - Analyze trends across sentiment and price changes
     
4. **News Summarization** 
    - Extract key positive and negative events from weekly news to provide concise insights.
    - Utilize a pre-trained large language model (model to be decided) to parse news and generate structured summaries.
    - Output: Weekly summaries with categorized positive and negative events.
    - Evaluation: Review summaries qualitatively for relevance and check their alignment with stock price trends.

---
