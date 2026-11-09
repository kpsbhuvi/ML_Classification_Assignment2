import os

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(PROJECT_DIR, "model")
RAW_DATA_FILE = os.path.join(PROJECT_DIR, "spambase.csv")
HOLDOUT_FILE = os.path.join(PROJECT_DIR, "test_data.csv")
LABEL_COLUMN = "label"

# display name -> (pickle file, whether the model expects scaled inputs)
MODEL_REGISTRY = {
    "Logistic Regression": ("logistic_regression.pkl", True),
    "Decision Tree": ("decision_tree.pkl", False),
    "KNN": ("knn.pkl", True),
    "Naive Bayes": ("naive_bayes.pkl", False),
    "Support Vector Machine": ("svm.pkl", True),
}

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def artifact_path(file_name):
    return os.path.join(ARTIFACT_DIR, file_name)


def score_predictions(y_true, y_pred, y_prob):
    """Return the six assignment metrics; weighted averaging for multi-class."""
    multiclass = len(np.unique(y_true)) > 2
    avg = "weighted" if multiclass else "binary"

    if multiclass:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
    else:
        auc = roc_auc_score(y_true, y_prob[:, 1])

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def load_artifacts():
    """Load the fitted preprocessing objects and all saved classifiers."""
    return {
        "scaler": joblib.load(artifact_path("scaler.pkl")),
        "feature_encoders": joblib.load(artifact_path("feature_encoders.pkl")),
        "target_encoder": joblib.load(artifact_path("target_encoder.pkl")),
        "models": {
            name: joblib.load(artifact_path(file_name))
            for name, (file_name, _) in MODEL_REGISTRY.items()
        },
    }


def encode_features(frame, feature_encoders, feature_order):
    """Apply the saved label encoders; unseen categories map to -1."""
    encoded = frame.copy()
    for col, encoder in feature_encoders.items():
        lookup = {label: idx for idx, label in enumerate(encoder.classes_)}
        encoded[col] = encoded[col].astype(str).map(lookup).fillna(-1).astype(int)
    return encoded[list(feature_order)]
