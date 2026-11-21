"""Configurable SVC; no probability claims from decision-function scores."""
from sklearn.svm import SVC

def make_svm(kernel: str = "rbf", C: float = 1.0, gamma: str | float = "scale", degree: int = 3, seed: int = 42) -> SVC:
    if kernel not in ("linear","rbf","poly") or C <= 0 or degree < 1:
        raise ValueError("Invalid SVM kernel, C, or degree")
    return SVC(kernel=kernel,C=C,gamma=gamma,degree=degree,random_state=seed)
