# Movie Genre Classifier

This project predicts the **genre of a movie** based on its **plot description**.

## Dataset
- Source: [Kaggle – Genre Classification Dataset (IMDb)](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)
- Place the dataset file (e.g., `train_data.txt`) inside this folder.

## Model
- Uses **TF-IDF vectorization** on text features.
- Classifier: **LinearSVC** (can also try Logistic Regression or Naive Bayes).

## How to Run
```bash
cd MovieGenreML
python movie_genre_classifier.py
