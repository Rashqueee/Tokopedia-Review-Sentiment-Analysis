# Tokopedia Review Sentiment Analysis

Proyek analisis sentimen untuk review produk Tokopedia menggunakan machine learning dan natural language processing (NLP).

## 📋 Deskripsi Proyek

Proyek ini bertujuan untuk menganalisis sentimen dari review pelanggan di Tokopedia. Dengan menggunakan teknik machine learning dan NLP, sistem ini dapat mengklasifikasikan review menjadi sentimen positif, negatif, atau netral.

## 🚀 Fitur

- **Web Scraping**: Mengambil data review dari Tokopedia menggunakan Selenium
- **Exploratory Data Analysis (EDA)**: Analisis dan visualisasi data review
- **Text Preprocessing**:  Pembersihan dan preprocessing teks Bahasa Indonesia
- **Sentiment Analysis**: Klasifikasi sentimen menggunakan machine learning
- **Model Training**: Pelatihan model dengan berbagai algoritma
- **Handling Imbalanced Data**: Menggunakan teknik oversampling/undersampling

## 📁 Struktur Proyek

```
Tokopedia-Review-Sentiment-Analysis/
│
├── data/                    # Dataset review yang telah di-scrape
├── models/                  # Model yang telah dilatih
├── notebooks/               # Jupyter notebooks untuk analisis
│   ├── 1_eda-and-cleaning.ipynb
│   ├── 2_preprocessing-and-modeling.ipynb
│   └── nltk_data/          # Data NLTK
├── scraper. py              # Script untuk scraping review
├── requirements.txt        # Dependencies Python
└── README.md              # Dokumentasi proyek
```

## 🛠️ Teknologi yang Digunakan

- **Python 3.x**
- **Selenium** - Web scraping
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning
- **NLTK** - Natural language processing
- **Sastrawi** - Indonesian stemming
- **Matplotlib & Seaborn** - Data visualization
- **WordCloud** - Visualisasi kata
- **Imbalanced-learn** - Handling imbalanced data

## 📦 Instalasi

1. Clone repository ini: 
```bash
git clone https://github.com/Rashqueee/Tokopedia-Review-Sentiment-Analysis.git
cd Tokopedia-Review-Sentiment-Analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (jika diperlukan):
```python
import nltk
nltk. download('punkt')
nltk.download('stopwords')
```

## 💻 Cara Menggunakan

### 1. Scraping Data Review

Jalankan script scraper untuk mengambil data review dari Tokopedia:

```bash
python scraper.py
```

### 2. Exploratory Data Analysis & Cleaning

Buka dan jalankan notebook pertama untuk analisis dan pembersihan data:

```bash
jupyter notebook notebooks/1_eda-and-cleaning.ipynb
```

### 3. Preprocessing & Modeling

Buka dan jalankan notebook kedua untuk preprocessing dan pelatihan model:

```bash
jupyter notebook notebooks/2_preprocessing-and-modeling.ipynb
```

## 📊 Tahapan Analisis

1. **Data Collection**: Web scraping review dari Tokopedia
2. **Data Cleaning**: Pembersihan data dari noise dan missing values
3. **Exploratory Data Analysis**: Visualisasi dan analisis distribusi data
4. **Text Preprocessing**:
   - Case folding
   - Tokenization
   - Stopwords removal
   - Stemming (Sastrawi)
5. **Feature Extraction**: TF-IDF / Bag of Words
6. **Model Training**:  Training dengan berbagai algoritma ML
7. **Model Evaluation**: Evaluasi performa model
8. **Model Deployment**: Menyimpan model terbaik

## 📈 Hasil

Detail hasil analisis dan performa model dapat dilihat di notebook yang tersedia di folder `notebooks/`.

## 🤝 Kontribusi

Kontribusi, issues, dan feature requests sangat diterima! 

## 📝 Lisensi

Proyek ini dibuat untuk keperluan pembelajaran dan penelitian. 

## 👤 Author

**RashkyJauhara**
- GitHub: [@Rashqueee](https://github.com/Rashqueee)

## 🙏 Acknowledgments

- Tokopedia sebagai sumber data
- Komunitas open source Python dan machine learning
