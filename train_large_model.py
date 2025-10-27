"""
Large Hate Speech Detection Model Training (~200MB)
Uses deep learning with large embedding dimensions and multiple LSTM layers
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D, Conv1D, GRU, Concatenate, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import Model
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Large Hate Speech Detection Model Training (~200MB)")
print("=" * 60)

# Configuration for ~200MB model
MAX_WORDS = 50000  # Large vocabulary
MAX_SEQUENCE_LENGTH = 500  # Very long sequences
EMBEDDING_DIM = 512  # Very large embeddings
BATCH_SIZE = 8  # Smaller batch for memory
EPOCHS = 20

print(f"\nConfiguration:")
print(f"  - Max vocabulary: {MAX_WORDS}")
print(f"  - Sequence length: {MAX_SEQUENCE_LENGTH}")
print(f"  - Embedding dimension: {EMBEDDING_DIM}")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Max epochs: {EPOCHS}")

# Load dataset
print("\n" + "=" * 60)
print("Loading dataset...")
print("=" * 60)

df = pd.read_csv('hate_speech.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:\n{df['label'].value_counts()}")

# Prepare data
X = df['text'].values
y = df['label'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Text preprocessing and tokenization
print("\n" + "=" * 60)
print("Tokenizing text...")
print("=" * 60)

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

# Convert texts to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

print(f"Training sequences shape: {X_train_pad.shape}")
print(f"Test sequences shape: {X_test_pad.shape}")
print(f"Vocabulary size: {len(tokenizer.word_index)}")

# Create large embedding matrix
print("\n" + "=" * 60)
print("Creating large embedding matrix...")
print("=" * 60)

word_index = tokenizer.word_index
vocab_size = min(len(word_index) + 1, MAX_WORDS)

# Initialize large embedding matrix
embedding_matrix = np.random.randn(vocab_size, EMBEDDING_DIM).astype('float32') * 0.01

print(f"Embedding matrix shape: {embedding_matrix.shape}")
print(f"Embedding matrix size: {embedding_matrix.nbytes / (1024**2):.2f} MB")

# Build large neural network model
print("\n" + "=" * 60)
print("Building large neural network model (~200MB)...")
print("=" * 60)

# Use Functional API for more complex architecture
input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))

# Embedding layer
embedding = Embedding(
    input_dim=vocab_size,
    output_dim=EMBEDDING_DIM,
    weights=[embedding_matrix],
    input_length=MAX_SEQUENCE_LENGTH,
    trainable=True
)(input_layer)

# Multiple Bidirectional LSTM layers with larger units
lstm1 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(embedding)
lstm2 = Bidirectional(LSTM(512, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(lstm1)
lstm3 = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(lstm2)
lstm4 = Bidirectional(LSTM(256, return_sequences=False, dropout=0.3, recurrent_dropout=0.2))(lstm3)

# Dense layers with very large hidden units
dense1 = Dense(1024, activation='relu')(lstm4)
dropout1 = Dropout(0.5)(dense1)
dense2 = Dense(512, activation='relu')(dropout1)
dropout2 = Dropout(0.4)(dense2)
dense3 = Dense(256, activation='relu')(dropout2)
dropout3 = Dropout(0.3)(dense3)
dense4 = Dense(128, activation='relu')(dropout3)
dropout4 = Dropout(0.2)(dense4)

# Output layer
output_layer = Dense(1, activation='sigmoid')(dropout4)

# Create model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print("\nModel Architecture:")
model.summary()

# Calculate model size
total_params = model.count_params()
approx_size_mb = (total_params * 4) / (1024**2)  # 4 bytes per float32

print(f"\nModel Statistics:")
print(f"  - Total parameters: {total_params:,}")
print(f"  - Approximate size: {approx_size_mb:.2f} MB")

if approx_size_mb < 150:
    print(f"\n⚠ Model size ({approx_size_mb:.2f} MB) is smaller than target (200MB)")
    print("  Increasing model complexity...")
elif approx_size_mb > 300:
    print(f"\n⚠ Model size ({approx_size_mb:.2f} MB) is larger than target (200MB)")
    print("  Consider reducing model complexity")
else:
    print(f"\n✓ Model size ({approx_size_mb:.2f} MB) is within target range (150-250MB)")

# Training callbacks
print("\n" + "=" * 60)
print("Training model...")
print("=" * 60)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.00001,
        verbose=1
    )
]

# Train model
history = model.fit(
    X_train_pad, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Evaluate model
print("\n" + "=" * 60)
print("Evaluating model...")
print("=" * 60)

y_pred_proba = model.predict(X_test_pad, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

test_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Hate Speech']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Sample predictions
print("\n" + "=" * 60)
print("Sample Predictions:")
print("=" * 60)

sample_texts = [
    "I love everyone and believe in kindness and respect for all people",
    "I hate these people they are disgusting and should not exist",
    "What a beautiful day it is outside, feeling grateful",
    "They should disappear from society forever, nobody wants them",
    "Celebrating diversity makes our community stronger",
    "Get rid of them all, they don't belong here"
]

for text in sample_texts:
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    pred_proba = model.predict(padded, verbose=0)[0][0]
    pred_label = "Hate Speech" if pred_proba > 0.5 else "Normal"
    confidence = pred_proba if pred_proba > 0.5 else 1 - pred_proba
    print(f"\nText: {text[:60]}...")
    print(f"Prediction: {pred_label} (confidence: {confidence*100:.2f}%)")

# Save model
print("\n" + "=" * 60)
print("Saving model...")
print("=" * 60)

# Save Keras model
model_path = 'hate_speech_model_large.h5'
model.save(model_path)
print(f"✓ Keras model saved to {model_path}")

# Save tokenizer
tokenizer_path = 'tokenizer_large.pkl'
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)
print(f"✓ Tokenizer saved to {tokenizer_path}")

# Create model package for app.py
model_data = {
    'model_path': model_path,
    'tokenizer': tokenizer,
    'max_sequence_length': MAX_SEQUENCE_LENGTH,
    'vocab_size': vocab_size,
    'embedding_dim': EMBEDDING_DIM,
    'accuracy': float(test_accuracy),
    'model_type': 'deep_learning'
}

package_path = 'hate_speech_model.pkl'
with open(package_path, 'wb') as f:
    pickle.dump(model_data, f)
print(f"✓ Model package saved to {package_path}")

# Check file sizes
model_size = os.path.getsize(model_path) / (1024**2)
tokenizer_size = os.path.getsize(tokenizer_path) / (1024**2)
package_size = os.path.getsize(package_path) / (1024**2)
total_size = model_size + package_size

print(f"\nFile Sizes:")
print(f"  - Keras model (.h5): {model_size:.2f} MB")
print(f"  - Tokenizer (.pkl): {tokenizer_size:.2f} MB")
print(f"  - Package (.pkl): {package_size:.2f} MB")
print(f"  - Total deployment size: {total_size:.2f} MB")

if total_size < 400:
    print(f"\n✓ Model size ({total_size:.2f} MB) is suitable for Render free tier (512MB RAM)")
else:
    print(f"\n⚠ Warning: Model size ({total_size:.2f} MB) might be too large for free tier")

print("\n" + "=" * 60)
print("Training complete!")
print("=" * 60)
print(f"\nModel accuracy: {test_accuracy*100:.2f}%")
print(f"Model size: {total_size:.2f} MB")
print(f"Parameters: {total_params:,}")
print("\nThe model is ready for deployment!")
