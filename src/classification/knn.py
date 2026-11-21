"""Educational Euclidean KNN with deterministic majority-vote ties."""
import numpy as np

class EducationalKNN:
    def __init__(self, n_neighbors: int = 5):
        if not isinstance(n_neighbors,int) or n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")
        self.n_neighbors = n_neighbors

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EducationalKNN":
        X, y = np.asarray(X,dtype=float), np.asarray(y)
        if X.ndim != 2 or len(X) != len(y) or len(X) < self.n_neighbors or not np.isfinite(X).all():
            raise ValueError("Invalid training matrix or too few neighbors")
        self.X_, self.y_ = X.copy(), y.copy()
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self,"X_"):
            raise ValueError("Fit the model first")
        X = np.atleast_2d(np.asarray(X,dtype=float))
        if X.shape[1] != self.X_.shape[1] or not np.isfinite(X).all():
            raise ValueError("Invalid inference features")
        predictions = []
        for row in X:
            distances = np.sqrt(np.sum((self.X_-row)**2,axis=1))
            neighbors = np.argsort(distances,kind="stable")[:self.n_neighbors]
            counts = [np.sum(self.y_[neighbors] == label) for label in self.classes_]
            predictions.append(self.classes_[np.argmax(counts)])
        return np.asarray(predictions)
