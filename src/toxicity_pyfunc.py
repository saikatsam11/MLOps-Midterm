"""MLflow pyfunc wrapper around the TF-IDF + logistic regression artifacts.

Bundles the word vectorizer, char vectorizer and classifier into one MLflow
model whose predict() takes a DataFrame with a `comment_text` column and
returns the toxicity probability for each row.
"""
import numpy as np
import joblib
import mlflow.pyfunc
from scipy.sparse import hstack

ARTIFACT_FILES = {
    "word_vectorizer": "word_tfidf_vectorizer.joblib",
    "char_vectorizer": "char_tfidf_vectorizer.joblib",
    "classifier": "toxicity_logistic_regression.joblib",
}


class ToxicityModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.word_vectorizer = joblib.load(context.artifacts["word_vectorizer"])
        self.char_vectorizer = joblib.load(context.artifacts["char_vectorizer"])
        self.classifier = joblib.load(context.artifacts["classifier"])
        self.positive_index = list(self.classifier.classes_).index(1)

    def predict(self, context, model_input, params=None):
        texts = model_input["comment_text"].fillna("").astype(str)
        features = hstack(
            [self.word_vectorizer.transform(texts), self.char_vectorizer.transform(texts)],
            format="csr",
            dtype=np.float32,
        )
        return self.classifier.predict_proba(features)[:, self.positive_index]
