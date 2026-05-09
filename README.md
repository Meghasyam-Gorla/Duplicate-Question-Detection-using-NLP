# Duplicate Pairs Recognition

A Machine Learning based NLP project that detects whether two questions are duplicates or not.

This project uses:
- Text preprocessing
- Feature engineering
- TF-IDF vectorization
- Fuzzy string matching
- XGBoost Classification

The model predicts whether two given questions convey the same semantic meaning. :contentReference[oaicite:0]{index=0}

---

# Features

- Advanced text preprocessing
- Contraction handling
- Stopword analysis
- Token-based similarity features
- FuzzyWuzzy similarity scores
- Longest common substring analysis
- TF-IDF vectorization
- XGBoost model prediction

---

# Tech Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- NLTK
- FuzzyWuzzy
- BeautifulSoup
- Distance

---

# Project Structure

```bash
DUPLICATE-PAIRS-RECOGNITION/
│
├── analysis1.ipynb
├── analysis2.ipynb
├── analysis3.ipynb
├── testing.ipynb
├── testing.py
├── app.py
├── requirements.txt
├── tfidf.pkl
├── xgb.pkl
├── train.csv
└── README.md
```

---

# Working Process

## 1. Text Preprocessing

The questions are cleaned using:
- Lowercasing
- Special character replacement
- HTML tag removal
- Contraction expansion
- Regex cleaning

---

## 2. Feature Engineering

The project creates multiple handcrafted NLP features such as:

### Basic Features
- Question length
- Number of words
- Common word count
- Word share ratio

### Token Features
- Common non-stopwords
- Common stopwords
- Common tokens
- First word match
- Last word match

### Length Features
- Absolute length difference
- Mean length
- Longest common substring ratio

### Fuzzy Features
- Fuzz ratio
- Partial ratio
- Token sort ratio
- Token set ratio

---

## 3. TF-IDF Vectorization

Questions are transformed into numerical vectors using a pretrained TF-IDF vectorizer.

---

## 4. Model Prediction

The processed features are passed into a trained XGBoost model which predicts:

- Duplicate
- Not Duplicate

---

# Installation

- Clone the repository
- Install dependencies
- Run the Project

---

# Required Files

Make sure these files are present:

- `tfidf.pkl`
- `xgb.pkl`

These contain:
- Trained TF-IDF vectorizer
- Trained XGBoost model

---

# Model Pipeline

```text
Input Questions
       ↓
Preprocessing
       ↓
Feature Engineering
       ↓
TF-IDF Transformation
       ↓
XGBoost Prediction
       ↓
Duplicate / Not Duplicate
```

---

# Future Improvements

- Add deep learning models
- Use Word2Vec/BERT embeddings
- Improve semantic understanding

---

# Author

Megha Syam

Data Science | Machine Learning | NLP

---
