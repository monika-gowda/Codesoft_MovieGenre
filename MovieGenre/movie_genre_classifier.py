import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# -----------------------------
# Load dataset and prepare
# -----------------------------
dataset_file = "train_data.txt"

plots = []
genres = []

with open(dataset_file, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split(":::")
        if len(parts) >= 4:
            genre = parts[2].strip()
            plot = parts[3].strip()
            plots.append(plot)
            genres.append(genre)

df = pd.DataFrame({'plot': plots, 'genre': genres})

# Clean text


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


df['clean_plot'] = df['plot'].apply(clean_text)
df['primary_genre'] = df['genre'].apply(lambda x: x.split('|')[0])

# Train model
X = df['clean_plot']
y = df['primary_genre']

tfidf = TfidfVectorizer(stop_words='english',
                        max_features=10000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(X)

model = LinearSVC()
model.fit(X_tfidf, y)

# -----------------------------
# Prediction loop
# -----------------------------
print("\n🎬 Movie Genre Predictor (Type 'exit' to quit)")

while True:
    user_input = input("\nEnter movie plot: ")
    if user_input.lower() == "exit":
        break
    clean = clean_text(user_input)
    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]
    print("Predicted Genre:", pred)
