# Sri Lankan Telecom Customer Churn Prediction

An ensemble learning project that analyzes anonymized telecom customer data, compares churn-prediction models, and provides an interactive Streamlit dashboard for results and individual customer predictions.

## Features

- Compares Random Forest, AdaBoost, Gradient Boosting, and Stacking classifiers
- Displays test-set metrics, confusion matrices, ROC curves, and feature importance
- Provides training and testing dataset summaries
- Predicts churn for individual customers using saved trained models
- Loads processed data and saved results directly from the project directories

## Project Structure

```text
app.py
requirements.txt
data/processed/
models/
notebooks/
results/
  figures/
  metrics/
```

- `app.py` - Streamlit dashboard
- `data/processed/` - Processed training, testing, feature, and label files
- `models/` - Saved ensemble models and model metadata
- `notebooks/` - Model development and experimentation notebooks
- `results/metrics/` - Model comparisons, classification reports, and feature importance
- `results/figures/` - Evaluation and comparison visualizations

## Requirements

- Python 3.10 or later recommended
- A virtual environment is recommended

## Installation

From the project root:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment and install the dependencies again.

## Run the Dashboard

```powershell
streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## Dashboard Pages

1. **Project Overview** - Dataset counts, churn rates, and model summary
2. **Dataset Analysis** - Training churn distribution and test-data preview
3. **Model Comparison** - Accuracy, precision, recall, F1-score, and ROC-AUC
4. **Model Evaluation** - Prediction preview and confusion-matrix interpretation
5. **Feature Importance** - Top features for the selected model
6. **Customer Prediction** - Select a customer and trained model to predict churn

## Notes

- Predictions are based on prepared model features and should not be treated as a sole basis for real-world decisions.
- The app searches saved result directories for comparison and prediction files, so preserve the existing `models/`, `data/processed/`, and `results/` structure.

## AI Disclosure

AI-assisted tools were used to support code organization, documentation, and project setup. Model results, implementation decisions, and project outputs have undergone human review.
