"""Bag-of-words + Multinomial Naive Bayes baseline."""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def create_pipeline() -> Pipeline:
    # Exactly the same model family and text vectorization as in the original notebook.
    return Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB()),
    ])
