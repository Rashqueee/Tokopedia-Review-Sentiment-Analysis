# Tokopedia Review Sentiment Analysis

This project is an end-to-end data science solution designed to scrape, analyze, and classify sentiments from product reviews on Tokopedia, one of Indonesia's largest e-commerce platforms. The goal is to understand customer feedback by processing text data and building machine learning models.

## 📋 Project Overview

The project pipeline consists of three main stages:
1.  **Data Collection**: Scraping user reviews from Tokopedia product pages using Selenium.
2.  **Data Preparation**: Cleaning and preprocessing the raw text data, specifically tailored for the Indonesian language.
3.  **Modeling**: Building and evaluating machine learning models to classify the sentiment of the reviews (e.g., Positive, Negative).

## 📂 Project Structure

- **`data/`**: Contains the datasets used in the project.
    - `raw/`: Stores the raw CSV files directly obtained from the scraper.
    - `processed/`: Stores cleaned and preprocessed data ready for modeling.
- **`notebooks/`**: Jupyter Notebooks containing the analysis logic.
    - `1_eda-and-cleaning.ipynb`: Exploratory Data Analysis (EDA) and initial data cleaning.
    - `2_preprocessing-and-modeling.ipynb`: Advanced text preprocessing (tokenization, stemming) and model training.
- **`models/`**: Saved machine learning models for future use.
- **`scraper.py`**: The Python script used to scrape reviews from Tokopedia.
- **`requirements.txt`**: List of Python dependencies required to run the project.

## 🛠️ Tech Stack

- **Python**: Core programming language.
- **Selenium**: Used for web scraping dynamic content from Tokopedia.
- **Pandas & NumPy**: For data manipulation and analysis.
- **Scikit-learn**: For building machine learning models.
- **NLTK & Sastrawi**: Natural Language Processing (NLP) libraries. `Sastrawi` is specifically used for stemming Indonesian text.
- **Matplotlib & Seaborn**: For data visualization.

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Rashqueee/Tokopedia-Review-Sentiment-Analysis.git
cd Tokopedia-Review-Sentiment-Analysis
```

### 2. Install Dependencies
Make sure you have Python installed. Then, install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Usage

**Scraping Data:**
To collect new data, run the scraper script.
*Note: You may need to update the product URL inside `scraper.py` to target a specific product.*
```bash
python scraper.py
```
This will save the scraped reviews into `data/raw/tokopedia_reviews.csv`.

**Running Analysis:**
Open the Jupyter notebooks to follow the analysis steps:
1.  Start with `notebooks/1_eda-and-cleaning.ipynb` to clean the raw data.
2.  Proceed to `notebooks/2_preprocessing-and-modeling.ipynb` to train and evaluate the sentiment models.

```bash
jupyter notebook
```

## 📊 Features

- **Automated Scraping**: The scraper handles pagination, expands "See More" buttons automatically, and filters reviews by star rating to ensure a balanced dataset.
- **Indonesian NLP**: specific preprocessing steps for Indonesian text, including stopword removal and stemming.
- **Visualizations**: Insightful charts showing rating distributions and word clouds of customer feedback.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improving the model accuracy or scraping efficiency, feel free to open an issue or submit a pull request.
