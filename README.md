# Predict Customer Churn

Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree — Udacity

## Project Description

This project applies clean code principles to a machine learning workflow that identifies credit card customers likely to churn. The implementation refactors an exploratory notebook into a production-ready Python library with logging and unit tests.

Two models are trained and compared:
- **Random Forest Classifier**
- **Logistic Regression**

## Files and Data Description

```
.
├── churn_library.py                  # Core ML library (EDA, feature engineering, training)
├── churn_script_logging_and_tests.py # Unit tests with logging
├── churn_notebook.ipynb              # Original exploratory notebook
├── guide.ipynb                       # Project guide
├── requirements.txt                  # Python dependencies
├── data/
│   └── bank_data.csv                 # Customer dataset
├── images/                           # EDA plots and model result images
├── logs/
│   └── churn_library.log             # Test run logs
└── models/
    ├── rfc_model.pkl                  # Saved Random Forest model
    └── logistic_model.pkl             # Saved Logistic Regression model
```

### `churn_library.py` Functions

| Function | Description |
|---|---|
| `import_data(pth)` | Loads CSV from the given path and returns a DataFrame |
| `perform_eda(df)` | Runs exploratory data analysis and saves figures to `images/` |
| `encoder_helper(df, category_lst, response)` | Encodes categorical columns as churn proportion per category |
| `perform_feature_engineering(df, response)` | Splits data into train/test sets and returns `X_train, X_test, y_train, y_test` |
| `classification_report_image(...)` | Generates classification reports for both models and saves them to `images/` |
| `feature_importance_plot(model, X_data, output_pth)` | Creates and saves a feature importance plot |
| `train_models(X_train, X_test, y_train, y_test)` | Trains both models, saves results to `images/` and model artifacts to `models/` |

## Running Files

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

- **Mac/Linux:** `source .venv/bin/activate`
- **Windows:** `.venv\Scripts\activate`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the churn library

```python
import churn_library as cls

df = cls.import_data("./data/bank_data.csv")
cls.perform_eda(df)

X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df, response='Churn')
cls.train_models(X_train, X_test, y_train, y_test)
```

### 4. Run tests and logging

```bash
pytest churn_script_logging_and_tests.py -v
```

Test results are written to `./logs/churn_library.log`. Each test function validates a step in the pipeline (data import, EDA, encoding, feature engineering, and model training).
