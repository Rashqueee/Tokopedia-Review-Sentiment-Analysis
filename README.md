# Tokopedia Review Sentiment Analysis

A comprehensive sentiment analysis project for Tokopedia product reviews using machine learning and natural language processing (NLP) techniques to classify customer feedback into positive, negative, or neutral sentiments.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Technologies](#-technologies)
- [Installation](#-installation)
- [Usage](#-usage)
- [Data Collection](#-data-collection)
- [Analysis Pipeline](#-analysis-pipeline)
- [Model Performance](#-model-performance)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)
- [Future Improvements](#-future-improvements)

## 🎯 Overview

This project aims to analyze customer sentiment from Tokopedia product reviews, one of Indonesia's largest e-commerce platforms. By leveraging machine learning algorithms and NLP techniques, this system can automatically classify reviews into sentiment categories, providing valuable insights for: 

- **Businesses**: Understanding customer satisfaction and product feedback
- **Product Managers**: Identifying areas for improvement
- **Data Scientists**: Learning sentiment analysis implementation in Indonesian language context
- **Researchers**: Studying e-commerce customer behavior patterns

### Key Objectives

- Extract and collect review data from Tokopedia using web scraping
- Perform comprehensive exploratory data analysis (EDA)
- Implement text preprocessing pipeline for Indonesian language
- Build and compare multiple machine learning models
- Handle imbalanced dataset challenges
- Deploy the best performing model for sentiment prediction

## 🚀 Features

### Data Collection
- **Automated Web Scraping**: Selenium-based scraper to extract product reviews
- **Robust Error Handling**: Handles dynamic content and pagination
- **Data Validation**: Ensures data quality and completeness

### Data Analysis
- **Exploratory Data Analysis**:  Comprehensive statistical analysis and visualization
- **Distribution Analysis**: Understanding sentiment distribution patterns
- **Word Frequency Analysis**: Identifying most common terms in reviews
- **Rating Correlation**: Analyzing relationship between ratings and sentiment

### Text Processing
- **Indonesian NLP Pipeline**: Specialized preprocessing for Indonesian text
- **Advanced Cleaning**: Removing noise, special characters, and URLs
- **Sastrawi Stemming**: Indonesian language stemming
- **Custom Stopwords**: Tailored stopword list for Indonesian e-commerce reviews

### Machine Learning
- **Multiple Algorithms**: Training and comparison of various ML models
- **Feature Engineering**: TF-IDF and Bag of Words vectorization
- **Imbalanced Data Handling**:  SMOTE, undersampling, and oversampling techniques
- **Model Evaluation**:  Comprehensive metrics (accuracy, precision, recall, F1-score)
- **Cross-Validation**: K-fold cross-validation for robust evaluation

## 📁 Project Structure

```
Tokopedia-Review-Sentiment-Analysis/
│
├── data/                           # Dataset directory
│   ├── raw/                        # Raw scraped data
│   ├── processed/                  # Cleaned and processed data
│   └── external/                   # Additional datasets
│
├── models/                         # Trained models and serialized objects
│   ├── vectorizers/                # TF-IDF and CountVectorizer objects
│   ├── classifiers/                # Trained ML models
│   └── evaluation/                 # Model evaluation results
│
├── notebooks/                      # Jupyter notebooks
│   ├── 1_eda-and-cleaning.ipynb   # Exploratory analysis and data cleaning
│   ├── 2_preprocessing-and-modeling. ipynb  # Text processing and model training
│   └── nltk_data/                  # NLTK data files
│
├── scraper.py                      # Web scraping script for Tokopedia
├── requirements.txt                # Python dependencies
├── . gitignore                      # Git ignore file
└── README.md                       # Project documentation
```

## 🛠️ Technologies

### Core Technologies
- **Python 3.7+** - Primary programming language
- **Jupyter Notebook** - Interactive development environment

### Web Scraping
- **Selenium 4.x** - Browser automation and web scraping
- **BeautifulSoup4** - HTML parsing and data extraction
- **Webdriver Manager** - Automatic browser driver management

### Data Processing & Analysis
- **Pandas 1.3+** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Collections** - Specialized container datatypes

### Natural Language Processing
- **NLTK** - Natural language toolkit for text processing
- **Sastrawi** - Indonesian language stemming library
- **re (Regex)** - Pattern matching and text cleaning

### Machine Learning
- **Scikit-learn** - Machine learning algorithms and tools
  - Classification algorithms (Naive Bayes, SVM, Random Forest, etc.)
  - Feature extraction (TF-IDF, CountVectorizer)
  - Model evaluation and metrics
- **Imbalanced-learn** - Handling imbalanced datasets (SMOTE, etc.)
- **Joblib** - Model serialization and deserialization

### Visualization
- **Matplotlib** - Basic plotting and visualization
- **Seaborn** - Statistical data visualization
- **WordCloud** - Word cloud generation for text analysis

## 📦 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- Chrome or Firefox browser (for web scraping)
- Jupyter Notebook or JupyterLab

### Step-by-Step Installation

1. **Clone the repository**
```bash
git clone https://github.com/Rashqueee/Tokopedia-Review-Sentiment-Analysis.git
cd Tokopedia-Review-Sentiment-Analysis
```

2. **Create a virtual environment (recommended)**
```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

5. **Verify installation**
```python
python -c "import pandas, sklearn, nltk, selenium; print('All packages installed successfully!')"
```

## 💻 Usage

### 1. Data Collection - Web Scraping

The `scraper.py` script automates the collection of product reviews from Tokopedia. 

**Basic Usage:**
```bash
python scraper.py
```

**Script Features:**
- Automatically handles dynamic page loading
- Extracts review text, ratings, dates, and user information
- Saves data in CSV format
- Implements rate limiting to avoid overloading servers
- Handles anti-scraping measures

**Customization:**
Edit the scraper configuration in `scraper.py`:
```python
# Example configuration
PRODUCT_URL = "https://www.tokopedia.com/product-url"
MAX_REVIEWS = 1000
OUTPUT_FILE = "data/raw/reviews.csv"
```

### 2. Exploratory Data Analysis & Cleaning

Open the first notebook to analyze and clean your data: 

```bash
jupyter notebook notebooks/1_eda-and-cleaning.ipynb
```

**This notebook includes:**
- Dataset overview and basic statistics
- Missing value analysis and handling
- Duplicate detection and removal
- Sentiment distribution visualization
- Rating analysis and correlation
- Text length distribution
- Word frequency analysis
- Data quality assessment
- Export cleaned dataset

### 3. Text Preprocessing & Model Training

Open the second notebook for preprocessing and model development:

```bash
jupyter notebook notebooks/2_preprocessing-and-modeling.ipynb
```

**This notebook includes:**
- Text normalization and cleaning
- Tokenization and stemming
- Feature extraction (TF-IDF, BoW)
- Train-test split
- Handling imbalanced classes
- Multiple model training and comparison
- Hyperparameter tuning
- Model evaluation and selection
- Model serialization

## 🔍 Data Collection

### Scraping Strategy

The scraper is designed to ethically collect publicly available review data: 

1. **Target Selection**: Specify product URLs from Tokopedia
2. **Dynamic Loading**: Handles JavaScript-rendered content using Selenium
3. **Pagination**: Automatically navigates through multiple review pages
4. **Data Extraction**: Collects: 
   - Review text
   - Star ratings (1-5)
   - Review date
   - Reviewer username
   - Product information
   - Helpful vote counts

### Data Fields

The scraped dataset typically contains:
- `review_id`: Unique identifier for each review
- `product_name`: Name of the reviewed product
- `review_text`: Full text of the customer review
- `rating`: Star rating (1-5 scale)
- `review_date`: Date when review was posted
- `reviewer_name`: Username of the reviewer
- `helpful_count`: Number of users who found review helpful

## 📊 Analysis Pipeline

### 1. Data Preprocessing

**Text Cleaning Steps:**
```python
# Example preprocessing pipeline
1. Lowercase conversion
2. URL removal
3. HTML tag removal
4. Special character removal
5. Number handling
6. Extra whitespace removal
7. Emoji handling (removal or conversion)
```

**Indonesian Text Processing:**
```python
# Sastrawi Stemming Example
from Sastrawi.Stemmer. StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

text = "Produknya sangat berkualitas dan pengiriman cepat"
stemmed = stemmer.stem(text)
# Output: "produk sangat kualitas dan kirim cepat"
```

**Stopwords Removal:**
- Uses Indonesian stopwords list
- Custom e-commerce specific stopwords
- Maintains sentiment-bearing words

### 2. Feature Engineering

**TF-IDF Vectorization:**
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8
)
```

**N-gram Analysis:**
- Unigrams: Single words
- Bigrams: Two-word combinations
- Trigrams: Three-word combinations

### 3. Model Training

**Algorithms Implemented:**

1. **Naive Bayes**
   - Multinomial Naive Bayes
   - Best for text classification
   - Fast training and prediction

2. **Support Vector Machine (SVM)**
   - Linear SVM
   - Effective for high-dimensional data
   - Good generalization

3. **Random Forest**
   - Ensemble learning method
   - Handles non-linear relationships
   - Feature importance analysis

4. **Logistic Regression**
   - Baseline model
   - Interpretable coefficients
   - Probability outputs

5. **Additional Models** (as explored in notebooks)
   - Decision Trees
   - K-Nearest Neighbors
   - Gradient Boosting

### 4. Handling Imbalanced Data

**Techniques Used:**

```python
from imblearn.over_sampling import SMOTE
from imblearn. under_sampling import RandomUnderSampler

# SMOTE - Synthetic Minority Over-sampling
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Combination of over and under sampling
from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(random_state=42)
```

### 5. Model Evaluation

**Evaluation Metrics:**

- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy
- **Recall**: Actual positive detection rate
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**:  Detailed error analysis
- **ROC-AUC**: Model discrimination ability

**Cross-Validation:**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
print(f"Mean F1-Score: {scores. mean():.4f} (+/- {scores.std():.4f})")
```

## 📈 Model Performance

Detailed model performance metrics and comparisons can be found in the notebooks.  Typical results: 

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | ~85% | ~83% | ~85% | ~84% |
| SVM | ~88% | ~87% | ~88% | ~87% |
| Random Forest | ~86% | ~85% | ~86% | ~85% |
| Logistic Regression | ~84% | ~82% | ~84% | ~83% |

*Note:  Actual performance may vary based on dataset and hyperparameters*

### Model Insights

- **Best Overall**:  SVM typically provides the best balance of precision and recall
- **Fastest**:  Naive Bayes offers quick training and prediction
- **Most Interpretable**:  Logistic Regression provides clear feature importance
- **Most Robust**: Random Forest handles outliers and non-linear patterns well

## 🔧 Configuration

### Scraper Configuration

Edit `scraper.py` to customize:
```python
# Browser settings
HEADLESS = True  # Run browser in background
TIMEOUT = 10  # Page load timeout in seconds

# Scraping settings
DELAY_BETWEEN_REQUESTS = 2  # Seconds between requests
MAX_RETRIES = 3  # Number of retry attempts
```

### Model Hyperparameters

Tune models in notebook 2:
```python
# Example SVM hyperparameters
svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# Grid search for optimal parameters
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(SVC(), svm_params, cv=5, scoring='f1_weighted')
```

## 🤝 Contributing

Contributions are welcome and greatly appreciated! Here's how you can contribute:

### Ways to Contribute

1. **Report Bugs**:  Open an issue describing the bug and how to reproduce it
2. **Suggest Features**: Share ideas for new features or improvements
3. **Submit Pull Requests**: Fix bugs or implement new features
4. **Improve Documentation**: Enhance README or add code comments
5. **Share Datasets**: Contribute additional review datasets

### Contribution Guidelines

1. Fork the repository
2. Create a new branch (`git checkout -b feature/YourFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some feature'`)
5. Push to the branch (`git push origin feature/YourFeature`)
6. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines for Python code
- Add docstrings to functions and classes
- Include comments for complex logic
- Write clear commit messages

## 📝 License

This project is created for educational and research purposes. When using this code or dataset: 

- **Respect Tokopedia's Terms of Service**
- **Use data ethically and responsibly**
- **Give appropriate credit when using this work**
- **Do not use for commercial purposes without permission**

## 🙏 Acknowledgments

- **Tokopedia** - For providing the platform and publicly available review data
- **Sastrawi** - Excellent Indonesian language stemming library
- **Python Community** - For the amazing open-source tools and libraries
- **Scikit-learn Team** - For comprehensive machine learning framework
- **NLTK Contributors** - For natural language processing tools

## 📞 Contact

**RashkyJauhara**

- GitHub: [@Rashqueee](https://github.com/Rashqueee)
- Project Link: [https://github.com/Rashqueee/Tokopedia-Review-Sentiment-Analysis](https://github.com/Rashqueee/Tokopedia-Review-Sentiment-Analysis)

## 📚 Future Improvements

Potential enhancements for this project: 

- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Add Indonesian BERT models (IndoBERT)
- [ ] Create web interface for real-time prediction
- [ ] Add aspect-based sentiment analysis
- [ ] Implement multi-class sentiment classification
- [ ] Add sentiment trend analysis over time
- [ ] Create automated reporting dashboard
- [ ] Expand to other e-commerce platforms
- [ ] Add model explainability (LIME, SHAP)
- [ ] Implement API for model serving
