"""
Train a lightweight hate speech detection model
This creates a small model suitable for free tier deployment
"""

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('hate_speech.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:\n{df['label'].value_counts()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], 
    df['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label']
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Create TF-IDF vectorizer (lightweight)
print("\nCreating TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=5000,  # Limit features for smaller model
    ngram_range=(1, 2),  # Unigrams and bigrams
    min_df=2,
    max_df=0.8,
    strip_accents='unicode',
    lowercase=True
)

# Fit and transform training data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")

# Train Logistic Regression model (fast and lightweight)
print("\nTraining Logistic Regression model...")
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    C=1.0,
    class_weight='balanced'
)

model.fit(X_train_tfidf, y_train)
print("Model trained successfully!")

# Evaluate
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*50}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Hate Speech']))

# Save model and vectorizer together
print("\nSaving model...")
model_data = {
    'model': model,
    'vectorizer': vectorizer
}

with open('hate_speech_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✓ Model saved to hate_speech_model.pkl")

# Test with sample predictions
print("\n" + "="*50)
print("Sample Predictions:")
print("="*50)

test_samples = [
    "I love everyone and believe in kindness",
    "I hate these people they are disgusting",
    "What a beautiful day it is outside",
    "They should disappear from society"
]

for text in test_samples:
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    proba = model.predict_proba(text_tfidf)[0]
    confidence = max(proba)
    label = "Hate Speech" if prediction == 1 else "Normal"
    
    print(f"\nText: {text[:60]}...")
    print(f"Prediction: {label} (confidence: {confidence:.2%})")

# Check model size
import os
model_size = os.path.getsize('hate_speech_model.pkl') / (1024 * 1024)
print(f"\n✓ Model file size: {model_size:.2f} MB")
print(f"✓ Memory efficient: {'YES ✓' if model_size < 50 else 'NO ✗'}")
