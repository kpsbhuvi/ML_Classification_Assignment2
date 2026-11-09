import io

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from ml_utils import (
    HOLDOUT_FILE,
    LABEL_COLUMN,
    METRIC_NAMES,
    MODEL_REGISTRY,
    encode_features,
    load_artifacts,
    score_predictions,
)

st.set_page_config(page_title="Spam E-mail Classifier", layout="wide")

st.title("Spambase - Spam vs Ham Classification Model Evaluator")
st.write("BITS Pilani - M.Tech AIML | Machine Learning Assignment 2")
st.write("Student: Panneer Selvam K | Roll No: 2026ak05425")
st.caption(
    "Upload a hold-out CSV (features + target column). Pre-trained models are loaded "
    "from disk and are NOT retrained on the uploaded data."
)


@st.cache_resource(show_spinner="Loading saved models...")
def get_artifacts():
    return load_artifacts()


def read_uploaded_csv(file_obj):
    # sep=None lets pandas sniff ',' vs ';'
    return pd.read_csv(io.BytesIO(file_obj.getvalue()), sep=None, engine="python")


def split_label(frame, label_encoder):
    """Return (feature_frame, integer labels); accepts 'yes'/'no' text or 0/1."""
    labels = frame[LABEL_COLUMN]
    if not pd.api.types.is_numeric_dtype(labels):
        known = set(label_encoder.classes_)
        bad = set(labels.astype(str)) - known
        if bad:
            raise ValueError(f"Unknown labels in '{LABEL_COLUMN}': {sorted(bad)}")
        labels = label_encoder.transform(labels.astype(str))
    return frame.drop(columns=[LABEL_COLUMN]), pd.Series(labels).astype(int).to_numpy()


def predict_with(name, artifacts, feature_frame):
    _, needs_scaling = MODEL_REGISTRY[name]
    model_input = (
        artifacts["scaler"].transform(feature_frame) if needs_scaling else feature_frame
    )
    model = artifacts["models"][name]
    return model.predict(model_input), model.predict_proba(model_input)


# ---- Sidebar ----
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload test CSV", type=["csv"])
chosen_model = st.sidebar.selectbox("Choose a model", list(MODEL_REGISTRY.keys()))

try:
    artifacts = get_artifacts()
except FileNotFoundError:
    st.error("Saved models not found in 'model/'. Run `python train_models.py` first.")
    st.stop()

if uploaded is not None:
    test_frame = read_uploaded_csv(uploaded)
    source_label = f"Uploaded file: {uploaded.name}"
else:
    try:
        test_frame = pd.read_csv(HOLDOUT_FILE)
        source_label = "Bundled hold-out file: test_data.csv"
    except FileNotFoundError:
        st.info("Upload a test CSV from the sidebar to begin.")
        st.stop()

st.subheader(source_label)
st.write(f"Rows: {len(test_frame)} | Columns: {test_frame.shape[1]}")
st.dataframe(test_frame.head())

expected_cols = list(artifacts["scaler"].feature_names_in_)
missing_cols = [c for c in expected_cols + [LABEL_COLUMN] if c not in test_frame.columns]
if missing_cols:
    st.error(f"Uploaded CSV is missing required columns: {missing_cols}")
    st.stop()

try:
    raw_features, y_true = split_label(test_frame, artifacts["target_encoder"])
except ValueError as err:
    st.error(str(err))
    st.stop()

feature_matrix = encode_features(raw_features, artifacts["feature_encoders"], expected_cols)

# Score every model once so the comparison table and the detail view agree
all_scores, all_preds = {}, {}
for model_name in MODEL_REGISTRY:
    preds, probs = predict_with(model_name, artifacts, feature_matrix)
    all_preds[model_name] = preds
    all_scores[model_name] = score_predictions(y_true, preds, probs)

st.subheader(f"Evaluation Metrics - {chosen_model}")
metric_cols = st.columns(len(METRIC_NAMES))
for col, metric in zip(metric_cols, METRIC_NAMES):
    col.metric(metric, f"{all_scores[chosen_model][metric]:.4f}")

left, right = st.columns(2)
y_pred = all_preds[chosen_model]
class_names = [str(c) for c in artifacts["target_encoder"].classes_]

with left:
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        confusion_matrix(y_true, y_pred),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.subheader("Classification Report")
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(4))

st.subheader("All Models on This Test Data")
comparison = pd.DataFrame(all_scores).T[METRIC_NAMES].round(4)
st.dataframe(comparison)
