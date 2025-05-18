# 📩 Spam Detector

## 🔍 Overview

This project is a machine learning-based spam detection system built with Python. It uses natural language processing (NLP) techniques to clean and process SMS messages, and employs a Random Forest classifier to accurately distinguish between spam and non-spam (ham) messages.

---

## 🚀 Features

* Text preprocessing with NLTK (stopword removal, tokenization, and punctuation filtering)
* Feature extraction using TF-IDF vectorization
* Random Forest classification for high-accuracy prediction
* Model evaluation using precision, recall, and F1-score
* Real-time prediction on custom input messages

---

## 🛠️ Tech Stack

* **Python** – Core language for scripting and model development
* **Pandas** – Data handling and preprocessing
* **Scikit-learn** – Model training, evaluation, and TF-IDF vectorization
* **NLTK** – Natural Language Toolkit for stopwords and text processing
* **Regular Expressions & String** – Text cleaning utilities

---

## 📊 Machine Learning Workflow

1. **Data Loading** – Reads a CSV dataset of SMS messages
2. **Data Cleaning** – Removes punctuation, stopwords, and tokenizes text
3. **Feature Extraction** – Converts cleaned text into numeric features using TF-IDF
4. **Label Encoding** – Converts spam/ham labels into numeric form
5. **Model Training** – Uses Random Forest classifier to train on the dataset
6. **Evaluation** – Outputs precision, recall, and F1-score metrics
7. **Custom Input Prediction** – Classifies new messages in real-time

---

## 🧪 Example

```python
text = ["Congratulations! You've won a $1000 Walmart gift card. Click here to claim now."]
```

**Output:**

```
Prediction: spam
```

---

## 📁 Dataset

The dataset used in this project is the **SMS Spam Collection** dataset, originally available from the UCI Machine Learning Repository. It includes 7,000+ labeled SMS messages.

---

## 📌 How to Run

1. Clone the repository: `https://github.com/1046prt/spam_detector`
2. Install required packages: `pip install -r requirements.txt`
3. Run the script: `python spam_detector.py`

---

## 📬 Future Improvements

* Web interface using Streamlit or Flask
* Model comparison (e.g., Logistic Regression, Naive Bayes)
* Save and load model using joblib
* Add visualization of model performance

---

