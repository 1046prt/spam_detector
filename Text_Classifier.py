import nltk
import pandas as pd
import string
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Load the dataset
pd.set_option('display.max_colwidth', 100)
messages = pd.read_csv('Spam_Detection/spam.csv', encoding="latin-1")

# Check the column names to ensure we drop the correct columns
print(messages.columns)

# We already know that the relevant columns are 'Category' and 'Message', so no need to drop others
messages.columns = ['label', 'text']  # Rename for clarity

# Check for null values
print('Number of nulls in label: {}'.format(messages['label'].isnull().sum()))
print('Number of nulls in text: {}'.format(messages['text'].isnull().sum()))

# Check value counts
print(messages['label'].value_counts())

# Define stopwords
stopwords = nltk.corpus.stopwords.words('english')

# Function to clean text
def cleantext(text):
    text = "".join([char for char in text if char not in string.punctuation])
    tokens = re.split('\\W+', text)
    text = [word for word in tokens if word.lower() not in stopwords and word]  # Remove empty strings
    return ' '.join(text)

# Vectorize the text
# Disable token_pattern warning by setting it to None (because we are using custom tokenizer)
tfidf = TfidfVectorizer(tokenizer=lambda x: cleantext(x).split(), token_pattern=None)
x_tfidf = tfidf.fit_transform(messages['text'])

# Create DataFrame from TF-IDF features
x_features = pd.DataFrame(x_tfidf.toarray())

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(messages['label'])

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x_features, y_encoded, test_size=0.2, random_state=42)

# Train the Random Forest model
rf = RandomForestClassifier(random_state=42)
rf_model = rf.fit(x_train, y_train)

# Make predictions
y_pred = rf_model.predict(x_test)

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred, pos_label=1)  
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)
print('Precision: {} / Recall: {} / F1-Score: {}'.format(round(precision, 3), round(recall, 3), round(f1, 3)))

# Test with a new example message
text = ["Congratulations! You've won a $1000 Walmart gift card. Click here to claim now."]
text_tfidf = tfidf.transform(text)
x_features = pd.DataFrame(text_tfidf.toarray())
y_pred = rf_model.predict(x_features)
predicted_label = label_encoder.inverse_transform(y_pred)  
print('Prediction for "{}": {}'.format(text[0], predicted_label[0]))
