# Machine Learning Assignment - 2: Spambase E-mail Classification

## a. Problem Statement

The objective of this assignment is to classify an e-mail as spam or ham (not spam) from word-frequency and character-frequency statistics extracted from the message text, using five different machine learning classifiers and comparing them on Accuracy, AUC, Precision, Recall, F1 and MCC.

## b. Dataset Description

- **Dataset Name:** Spambase
- **Source:** UCI Machine Learning Repository - https://archive.ics.uci.edu/dataset/94/spambase
- **Instances:** 4,601 (1,813 spam, 2,788 ham; ~39% spam)
- **Features:** 57 numeric input features: 48 `word_freq_*` (percentage of words matching a given word), 6 `char_freq_*` (percentage of characters matching a given character), and 3 capital-letter run-length statistics (`capital_run_length_average`, `capital_run_length_longest`, `capital_run_length_total`)
- **Target:** Binary (`label`: `spam` = 1, `ham` = 0)
- **Missing values:** none
- **Train/Test split:** 80/20 stratified (`random_state=42`); the 921 test rows are saved as `test_data.csv`

## Repository Structure

```
app.py              Streamlit app (loads saved models, evaluates uploaded test CSV)
ml_utils.py         Shared paths, model registry, metric and encoding helpers
train_models.py     Trains the 5 models, saves them to model/, writes test_data.csv
spambase.csv        Full dataset used for training
test_data.csv       Hold-out test data (921 rows) to upload in the app
requirements.txt
model/              *.pkl: 5 classifiers + scaler + label encoder
```

## How to Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_models.py        # regenerates model/ and test_data.csv
streamlit run app.py
```

In the app, upload `test_data.csv` (or use the bundled default), pick a model, and view the metrics, confusion matrix and classification report.

## c. GitHub Repository Link

https://github.com/kpsbhuvi/ML_Classification_Assignment2.git

**Live Streamlit App:** https://mlclassificationassignment2-cvnbijesbym4bcovnkpmku.streamlit.app/

## d. Models Used & Evaluation Metrics

### Comparison Table

| ML Model Name              | Accuracy |  AUC   | Precision | Recall |   F1   |  MCC   |
| :------------------------- | :------: | :----: | :-------: | :----: | :----: | :----: |
| **Logistic Regression**    |  0.9294  | 0.9702 |  0.9209   | 0.8981 | 0.9093 | 0.8518 |
| **Decision Tree**          |  0.9034  | 0.9266 |  0.9308   | 0.8154 | 0.8693 | 0.7976 |
| **KNN**                    |  0.9077  | 0.9506 |  0.8861   | 0.8788 | 0.8824 | 0.8065 |
| **Naive Bayes**            |  0.8339  | 0.9449 |  0.7178   | 0.9532 | 0.8189 | 0.6941 |
| **Support Vector Machine** |  0.9273  | 0.9667 |  0.9277   | 0.8843 | 0.9055 | 0.8471 |

### Observations on Model Performance

The classes are reasonably balanced (~39% spam), so all six metrics are meaningful here. Spam is the positive class: Precision measures how many flagged e-mails really are spam, and Recall measures how much spam is caught.

| ML Model Name | Observation about model performance |
| :------------ | :---------------------------------- |
| **Logistic Regression** | Best on Accuracy, AUC, F1 and MCC. Once scaled, the word-frequency features are close to linearly separable, so a simple linear model generalises very well. |
| **Decision Tree** | Highest Precision (0.9308) but low Recall (0.8154), so it lets more spam through. The depth-limited tree (max_depth=6) is easy to interpret but less accurate. |
| **KNN** | Solid middle performer (Accuracy 0.9077, AUC 0.9506). Depends on feature scaling and is affected by the many sparse, high-dimensional features. |
| **Naive Bayes** | Highest Recall (0.9532) but the lowest Precision (0.7178), Accuracy (0.8339) and MCC (0.6941): it catches almost all spam but flags many genuine mails. The feature-independence assumption is violated by correlated word counts. |
| **Support Vector Machine** | Very close second: Accuracy 0.9273, AUC 0.9667, Precision 0.9277, MCC 0.8471. The RBF kernel captures non-linear patterns but is slower and gives no gain over Logistic Regression here. |
| **Overall Winner for your dataset?** | **Logistic Regression**: highest Accuracy (0.9294), AUC (0.9702), F1 (0.9093) and MCC (0.8518), with SVM a close runner-up. |
