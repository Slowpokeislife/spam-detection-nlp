# SMS Spam Detection | NLP & Machine Learning

A reproducible baseline for classifying English SMS messages as **spam** or **ham** (legitimate messages). The project refactors an exploratory Jupyter notebook into reusable data, model-training and inference modules.

## Project overview

- **Task:** binary text classification (ham = 0, spam = 1)
- **Text representation:** `CountVectorizer` (bag of words)
- **Classifier:** `MultinomialNB` (Multinomial Naive Bayes)
- **Evaluation:** stratified 80/20 train/test split, classification report and confusion matrix
- **Dataset:** 5,572 messages (4,825 ham / 747 spam), as recorded in the supplied notebook

### Original notebook results (historical)

| Metric | Value reported in original run |
|---|---:|
| Accuracy | ~0.99 |
| Spam precision | 0.98 |
| Spam recall | 0.93 |
| Spam F1-score | 0.95 |
| Test messages | 1,115 |

These figures are **from the provided notebook**, which used a random train/test split **without a fixed seed**. They are not claimed as measurements of the newly refactored scripts. Run `python -m src.train` to generate your own reproducible result with `random_state=42` and stratification. This simple baseline should not be described as production-grade spam protection.

## Repository structure

```text
spam-detection-nlp/
├── data/raw/              # Place downloaded spam.csv here (ignored by git)
├── notebooks/             # Clean exploratory notebook
├── src/
│   ├── data.py            # Dataset loading and validation
│   ├── model.py           # CountVectorizer + MultinomialNB pipeline
│   ├── train.py           # Training, evaluation and saving
│   └── predict.py         # Command-line predictions
├── tests/                 # Small unit tests with synthetic samples
├── models/                # Local trained artifacts (ignored by git)
├── reports/               # Generated metrics (ignored by git)
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

Source: [spam.csv by tmehul on Kaggle](https://www.kaggle.com/datasets/tmehul/spamcsv).

1. Download the dataset from Kaggle.
2. Place the CSV at `data/raw/spam.csv`.
3. Confirm the file has the columns **`Category`** and **`Message`**, as used by the original notebook.

The dataset is deliberately **not redistributed in this repository**. Consult its Kaggle page for applicable reuse terms.

## Setup

Python 3.10+ is recommended. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train and evaluate

```bash
python -m src.train
```

This writes `models/spam_pipeline.joblib` and `reports/metrics.json`. Both are intentionally untracked, so anyone can recreate them with the original dataset. The JSON contains class-level metrics and a confusion matrix in label order `[ham, spam]`.

## Predict new messages

```bash
python -m src.predict "Hey, are we meeting tomorrow?" "Congratulations! Claim your free prize now"
```

Shows predicted label and estimated spam probability for each message. Probabilities are model outputs, not guaranteed calibrated risk scores. Messages in languages or contexts unlike the training data may perform worse.

## Tests

```bash
pytest -q
```

Tests use synthetic messages and **do not** require downloading Kaggle data.

## How it works

1. Load and validate labeled SMS texts.
2. Convert `Category` into a numeric target (`ham=0`, `spam=1`).
3. Split data into train and test sets.
4. Fit bag-of-words vocabulary on the **training split only**.
5. Train Multinomial Naive Bayes and evaluate held-out predictions.
6. Save a single pipeline so inference applies the same vocabulary automatically.

## What I learned

- How sparse bag-of-words features encode text for classical ML models.
- Why feature extraction must be fit on training data to avoid data leakage.
- How Naive Bayes provides a fast, interpretable baseline for SMS classification.
- Why precision, recall and F1 for the minority **spam** class are more informative than accuracy alone.
- How to convert a research notebook into testable, reproducible inference code.

## Limitations and next steps

- The original notebook did not set a random seed; its exact results may not reproduce.
- The dataset is imbalanced (ham: 4,825; spam: 747); monitor spam recall and false positives.
- The baseline uses token counts rather than contextual semantics.
- Future upgrades: compare TF-IDF + logistic regression, analyze false positives/negatives, add cross-validation and a Streamlit demo.
- Do not use this model to make automated high-stakes decisions.

## Attribution

Dataset: [tmehul / spam.csv on Kaggle](https://www.kaggle.com/datasets/tmehul/spamcsv). Notebook and refactoring are included for portfolio and educational use.
