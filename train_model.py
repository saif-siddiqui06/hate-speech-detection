import pickle
import string
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


DATASET_PATH = Path("Dataset---Hate-Speech-Detection-using-Deep-Learning.csv")
MODEL_PATH = Path("Hate Speech") / "hate_speech_rnn_model.h5"
TOKENIZER_PATH = Path("Hate Speech") / "tokenizer.pkl"
MAX_WORDS = 10000
MAX_LEN = 100


nltk.download("stopwords")
nltk.download("wordnet")

df = pd.read_csv(DATASET_PATH)

class_0 = df[df["class"] == 0]
class_1 = df[df["class"] == 1].sample(n=3500, random_state=42)
class_2 = df[df["class"] == 2]
balanced_df = pd.concat([class_0, class_0, class_0, class_1, class_2], axis=0)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(words)


balanced_df["tweet"] = balanced_df["tweet"].apply(preprocess_text)

X = balanced_df["tweet"]
y = pd.get_dummies(balanced_df["class"])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True, split=" ")
tokenizer.fit_on_texts(X_train)

X_train_padded = pad_sequences(
    tokenizer.texts_to_sequences(X_train),
    maxlen=MAX_LEN,
    padding="post",
    truncating="post",
)
X_val_padded = pad_sequences(
    tokenizer.texts_to_sequences(X_val),
    maxlen=MAX_LEN,
    padding="post",
    truncating="post",
)

model = Sequential(
    [
        Embedding(input_dim=MAX_WORDS, output_dim=32, input_length=MAX_LEN),
        Bidirectional(LSTM(16)),
        Dense(512, activation="relu", kernel_regularizer="l1"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(3, activation="softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

callbacks = [
    EarlyStopping(patience=3, monitor="val_accuracy", restore_best_weights=True),
    ReduceLROnPlateau(patience=2, monitor="val_loss", factor=0.5),
]

model.fit(
    X_train_padded,
    y_train,
    validation_data=(X_val_padded, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
model.save(MODEL_PATH)
with TOKENIZER_PATH.open("wb") as file:
    pickle.dump(tokenizer, file)

print(f"Model saved to {MODEL_PATH}")
print(f"Tokenizer saved to {TOKENIZER_PATH}")
