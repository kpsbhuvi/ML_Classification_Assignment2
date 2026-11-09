import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml_utils import (
    ARTIFACT_DIR,
    HOLDOUT_FILE,
    LABEL_COLUMN,
    METRIC_NAMES,
    MODEL_REGISTRY,
    RAW_DATA_FILE,
    artifact_path,
    score_predictions,
)

SEED = 42


def build_classifiers():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(probability=True, random_state=SEED),
    }


def run_training():
    if not os.path.exists(RAW_DATA_FILE):
        print(f"Dataset not found: {RAW_DATA_FILE}")
        return

    raw = pd.read_csv(RAW_DATA_FILE)
    print(f"Loaded dataset, shape: {raw.shape}")

    label_col = LABEL_COLUMN if LABEL_COLUMN in raw.columns else raw.columns[-1]
    print(f"Label column: '{label_col}'")

    features = raw.drop(columns=[label_col])
    labels = raw[label_col]

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    label_encoder = LabelEncoder()
    labels_enc = label_encoder.fit_transform(labels.astype(str))
    joblib.dump(label_encoder, artifact_path("target_encoder.pkl"))

    category_encoders = {}
    for col in features.select_dtypes(include=["object", "category"]).columns:
        enc = LabelEncoder()
        features[col] = enc.fit_transform(features[col].astype(str))
        category_encoders[col] = enc
    joblib.dump(category_encoders, artifact_path("feature_encoders.pkl"))

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels_enc, test_size=0.2, random_state=SEED, stratify=labels_enc
    )

    # Hold-out rows in their original (un-encoded) form, for the Streamlit upload
    raw.loc[X_test.index].to_csv(HOLDOUT_FILE, index=False)
    print(f"Wrote hold-out file with {len(X_test)} rows -> {HOLDOUT_FILE}")

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    joblib.dump(scaler, artifact_path("scaler.pkl"))

    header = f"{'ML Model Name':<23}" + "".join(f" | {m:<9}" for m in METRIC_NAMES)
    print("\n" + header)
    print("-" * len(header))

    for name, clf in build_classifiers().items():
        file_name, needs_scaling = MODEL_REGISTRY[name]
        train_in = X_train_std if needs_scaling else X_train
        test_in = X_test_std if needs_scaling else X_test

        clf.fit(train_in, y_train)
        scores = score_predictions(y_test, clf.predict(test_in), clf.predict_proba(test_in))

        print(f"{name:<23}" + "".join(f" | {scores[m]:<9.4f}" for m in METRIC_NAMES))
        joblib.dump(clf, artifact_path(file_name))

    print(f"\nSaved 5 models and preprocessing artifacts to '{ARTIFACT_DIR}'")


if __name__ == "__main__":
    run_training()
