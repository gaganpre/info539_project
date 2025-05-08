# INFO 539 Project Repository

## NLP-Powered Financial News Analysis for Market Insights


# Project: NLP-Powered Financial News Analysis for Market Insights

## Motivation
The prices of stocks listed under global exchanges are influenced by a range of factors, with financial performance, innovation, partnerships, and market sentiment playing key roles. In the fast-paced financial industry, news and media coverage can significantly impact investor perception and, in turn, stock price movement. Given the overwhelming volume of news articles and social commentary, investment firms face increasing challenges in staying informed and responding rapidly.

## Objective
The goal of this project is to build an NLP system that:
* Analyzes the sentiment of financial news articles.
* Summarizes key positive and negative events from weekly news.
* Explores the correlation between news sentiment and stock market trends.


## Dataset
The project utilizes the following datasets:
* Stock News: [stock_news.csv](https://github.com/ArtZaragozaGitHub/NLP--P6_Sentiment_Analysis_and_Summarization_of_Stock_News/blob/main/stock_news.csv)
* Stock Tweets and Finance Data: [stock_yfinance_data.csv](https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction/data?select=stock_yfinance_data.csv)

*(Note: The primary dataset used in notebook is `stock_news.csv`)*

## Installation
All required packages are listed in `requirements.txt`. Install them using pip:
```bash
pip install -r requirements.txt
```

If you are using *uv* as a package manager:

```bash
pip install uv
uv sync 
```

Downloading glove:
```bash
# Run only once 
# Download Glove word embeddings model
# if glove.6B.zip is not present in the current directory
   if [ ! -f glove.6B.zip ]
   then
      echo "Glove model file not found! downloading..";
      wget http://nlp.stanford.edu/data/glove.6B.zip; unzip glove.6B.zip;
      else echo "File found! skipping download..."
   fi
   ls -l glove.*
```


## Usage
The Jupyter notebook `project.ipynb` demonstrates the following workflow:

1.  **Import Libraries**: Necessary libraries like pandas, numpy, matplotlib, seaborn, ydata_profiling, gensim, sentence_transformers, sklearn, torch, and tqdm are imported.
2.  **Load Data**: The `stock_news.csv` dataset is loaded into a pandas DataFrame.
3.  **Data Preprocessing**:
    * Checks data types and converts the 'Date' column to datetime objects.
    * Checks for and handles missing values and duplicates (though none were found in this dataset).
4.  **Exploratory Data Analysis (EDA)**:
    * Analyzes the distribution of sentiment labels (-1: Negative, 0: Neutral, 1: Positive).
    * Examines the distribution of stock prices (Open, High, Low, Close) using KDE plots.
    * Uses ydata-profiling for a detailed report.
    * Analyzes the length (word count) of news articles.
    * Analyzes the distribution of trading volume.
    * Performs bivariate analysis, including a correlation matrix and boxplots of labels vs. price/volume.
    * Analyzes price trends over time.
5.  **Train-Test-Validation Split**: Splits the data chronologically into training (until 2019-04-01), validation (2019-04-01 to 2019-04-16), and test (from 2019-04-16) sets.
6.  **Word Embeddings**: Generates text embeddings using:
    * Word2Vec (trained on the dataset).
    * GloVe (using pre-trained `glove.6B.100d.txt` embeddings).
    * Sentence Transformer (using the `all-MiniLM-L6-v2` model).
7.  **Sentiment Analysis (Classification)**:
    * Trains baseline classification models (Gradient Boosting is shown, but Decision Tree and Random Forest are mentioned) on each embedding type.
    * Evaluates baseline models using confusion matrices and classification reports.
    * Performs hyperparameter tuning using GridSearchCV for the Gradient Boosting classifier with each embedding type.
    * Evaluates the tuned models.
8.  **Model Selection**: Selects the best performing model based on evaluation metrics (Tuned GloVe + Gradient Boosting was chosen, based on validation performance).
9.  **Weekly News Summarization**:
    * Aggregates news articles on a weekly basis.
    * Uses a Large Language Model (Mistral-7B-Instruct-v0.2-GGUF) to summarize the top 3 positive and negative weekly events affecting stock performance.
    * Parses the JSON output from the LLM.


## Project Proposal
Read the full proposal [here](https://github.com/gaganpre/info539_project/blob/main/PROPOSAL.md).

## Project Repo :
You can find the full project on GitHub: [info539_project](https://github.com/gaganpre/info539_project)

## Project Notbook :
https://gaganpre.github.io/info539_project/project.html




