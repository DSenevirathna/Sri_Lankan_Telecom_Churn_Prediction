
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="wide"
)

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_DIR / "models"
METRICS_DIR = PROJECT_DIR / "results" / "metrics"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

MODEL_FILES = {
    "Random Forest": MODEL_DIR / "random_forest.joblib",
    "AdaBoost": MODEL_DIR / "adaboost.joblib",
    "Gradient Boosting": MODEL_DIR / "gradient_boosting.joblib",
    "Stacking": MODEL_DIR / "stacking.joblib",
}

# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def find_csv_file(directory, keywords):
    """Find a CSV file using possible filename keywords."""

    if not directory.exists():
        return None

    for file in directory.rglob("*.csv"):
        filename = file.name.lower()

        if all(word.lower() in filename for word in keywords):
            return file

    return None


def find_parquet_file(directory, filename):
    """Find a Parquet file by its exact filename."""

    path = directory / filename

    return path if path.exists() else None


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


@st.cache_data
def load_parquet(path):
    return pd.read_parquet(path)


@st.cache_resource
def load_models():
    models = {}

    for name, path in MODEL_FILES.items():

        if path.exists():
            models[name] = joblib.load(path)

    return models


def find_column(data, possible_names):
    """Find a column without depending on capitalization."""

    normalized = {
        str(col).lower().replace(" ", "_"): col
        for col in data.columns
    }

    for name in possible_names:
        if name.lower() in normalized:
            return normalized[name.lower()]

    return None


def show_metric_card(label, value):
    st.metric(label=label, value=value)


# --------------------------------------------------
# 3. LOAD SAVED RESULTS
# --------------------------------------------------

models = load_models()

comparison_file = find_csv_file(
    METRICS_DIR,
    ["test", "comparison"]
)

if comparison_file is None:
    comparison_file = find_csv_file(
        METRICS_DIR,
        ["model", "comparison"]
    )

comparison = None

if comparison_file:
    comparison = load_csv(comparison_file)


# --------------------------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("📊 Telecom Churn Dashboard")

st.sidebar.write(
    "Sri Lankan Telecom Customer Churn Prediction"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Dataset Analysis",
        "Model Comparison",
        "Model Evaluation",
        "Feature Importance",
        "Customer Prediction"
    ]
)

st.sidebar.divider()

st.sidebar.write("**Ensemble Learning Models**")

for model_name in MODEL_FILES:
    if model_name in models:
        st.sidebar.success(model_name)
    else:
        st.sidebar.warning(f"{model_name}: Not found")


# --------------------------------------------------
# 5. PROJECT OVERVIEW
# --------------------------------------------------

if page == "Project Overview":

    st.title("📊 Telecom Customer Churn Prediction")

    st.markdown("""
    ### Ensemble Learning Dashboard

    This project uses ensemble learning techniques to predict
    customer churn using anonymized telecom usage data.

    The objective is to compare different ensemble approaches
    and identify the most suitable model for customer churn prediction.
    """)

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    train_labels_path = PROCESSED_DIR / "y_train.csv"
    test_labels_path = PROCESSED_DIR / "y_test.csv"

    if train_labels_path.exists():
        train_labels = load_csv(train_labels_path)
        churn_col = find_column(
            train_labels, ["churn", "target", "label"]
        )

        if churn_col:
            total_train = len(train_labels)
            churn_count = int(train_labels[churn_col].sum())
            churn_rate = (
                churn_count / total_train * 100
                if total_train else 0
            )

            col1.metric("Training Customers", f"{total_train:,}")
            col2.metric("Training Churn Rate", f"{churn_rate:.2f}%")
            col3.metric("Churned Customers", f"{churn_count:,}")

    if test_labels_path.exists():
        test_labels = load_csv(test_labels_path)
        col4.metric("Testing Customers", f"{len(test_labels):,}")

    st.divider()

    st.subheader("Models Used")

    model_cols = st.columns(4)

    model_descriptions = [
        ("Random Forest", "Bagging-based ensemble learning."),
        ("AdaBoost", "Boosting-based ensemble learning."),
        ("Gradient Boosting", "Sequential boosting algorithm."),
        ("Stacking", "Combines multiple base models using a meta-model.")
    ]

    for col, (name, description) in zip(
        model_cols, model_descriptions
    ):
        with col:
            st.markdown(f"### {name}")
            st.write(description)

    st.info(
        "The dashboard compares the models using their "
        "measured test-set results."
    )


# --------------------------------------------------
# 6. DATASET ANALYSIS
# --------------------------------------------------

elif page == "Dataset Analysis":

    st.title("📁 Dataset Analysis")

    train_labels_path = PROCESSED_DIR / "y_train.csv"
    test_labels_path = PROCESSED_DIR / "y_test.csv"

    if train_labels_path.exists():

        train_labels = load_csv(train_labels_path)

        churn_col = find_column(
            train_labels, ["churn", "target", "label"]
        )

        if churn_col:

            st.subheader("Training Dataset Churn Distribution")

            distribution = (
                train_labels[churn_col]
                .value_counts()
                .rename_axis("Churn")
                .reset_index(name="Customers")
            )

            distribution["Churn"] = distribution["Churn"].map(
                {0: "Not Churned", 1: "Churned"}
            ).fillna(distribution["Churn"].astype(str))

            fig = px.pie(
                distribution,
                names="Churn",
                values="Customers",
                title="Customer Churn Distribution",
                hole=0.4
            )

            st.plotly_chart(fig, width="stretch")

            st.subheader("Training Dataset Statistics")

            st.dataframe(
                train_labels.describe(),
                width="stretch"
            )

    if test_labels_path.exists():

        test_labels = load_csv(test_labels_path)

        st.subheader("Testing Dataset")

        st.write(
            f"Number of testing customers: {len(test_labels):,}"
        )

        st.dataframe(
            test_labels.head(10),
            width="stretch"
        )

    else:
        st.warning("Processed label files were not found.")


# --------------------------------------------------
# 7. MODEL COMPARISON
# --------------------------------------------------

elif page == "Model Comparison":

    st.title("📈 Ensemble Model Comparison")

    if comparison is None:

        st.warning(
            "Test comparison CSV not found. "
            "Check results/metrics and your training notebook."
        )

    else:

        st.subheader("Test Set Performance")

        st.dataframe(
            comparison,
            width="stretch",
            hide_index=True
        )

        metric_aliases = {
            "accuracy": ["accuracy"],
            "precision": ["precision"],
            "recall": ["recall"],
            "f1": ["f1", "f1_score", "f1-score"],
            "roc_auc": ["roc_auc", "roc-auc", "auc"]
        }

        model_col = find_column(
            comparison, ["model", "model_name", "classifier"]
        )

        selected_metrics = []

        for metric, aliases in metric_aliases.items():
            col = find_column(comparison, aliases)

            if col:
                selected_metrics.append((metric, col))

        if model_col and selected_metrics:

            chart_data = comparison.melt(
                id_vars=[model_col],
                value_vars=[col for _, col in selected_metrics],
                var_name="Metric",
                value_name="Score"
            )

            fig = px.bar(
                chart_data,
                x=model_col,
                y="Score",
                color="Metric",
                barmode="group",
                title="Comparison of Ensemble Models",
                text_auto=".3f"
            )

            fig.update_layout(
                yaxis_range=[0, 1],
                xaxis_title="Ensemble Model",
                yaxis_title="Score"
            )

            st.plotly_chart(fig, width="stretch")

            st.subheader("Understanding the Metrics")

            st.markdown("""
            - **Accuracy:** Overall proportion of correct predictions.
            - **Precision:** Of customers predicted to churn, how many actually churned?
            - **Recall:** Of all customers who churned, how many did the model identify?
            - **F1-score:** Balance between precision and recall.
            - **ROC-AUC:** How well the model separates the two classes.
            """)

            st.info(
                "For churn prediction, recall and F1-score are "
                "particularly important because missing an actual "
                "churning customer can be costly."
            )


# --------------------------------------------------
# 8. MODEL EVALUATION
# --------------------------------------------------

elif page == "Model Evaluation":

    st.title("🎯 Model Evaluation")

    PREDICTIONS_DIR = PROJECT_DIR / "results"

    prediction_files = list(
      PREDICTIONS_DIR.glob("*test_predictions.csv")
    )

    if not prediction_files:

        st.warning(
            "Prediction CSV files were not found in results/metrics."
        )

    else:

        available_files = {
            file.stem: file for file in prediction_files
        }

        selected_file = st.selectbox(
            "Select prediction result",
            list(available_files.keys())
        )

        prediction_data = load_csv(
            available_files[selected_file]
        )

        st.subheader("Prediction Results")

        st.dataframe(
            prediction_data.head(20),
            width="stretch"
        )

        actual_col = find_column(
            prediction_data,
            ["actual_churn", "actual", "y_true", "churn"]
        )

        predicted_col = find_column(
            prediction_data,
            ["predicted_churn", "predicted", "y_pred", "prediction"]
        )

        if actual_col and predicted_col:

            actual = prediction_data[actual_col]
            predicted = prediction_data[predicted_col]

            cm = confusion_matrix(
                actual,
                predicted,
                labels=[0, 1]
            )

            fig = px.imshow(
                cm,
                text_auto=True,
                labels={
                    "x": "Predicted Class",
                    "y": "Actual Class",
                    "color": "Customers"
                },
                x=["Not Churned", "Churned"],
                y=["Not Churned", "Churned"],
                title="Confusion Matrix"
            )

            st.plotly_chart(fig, width="stretch")

            tn, fp, fn, tp = cm.ravel()

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("True Negatives", tn)
            c2.metric("False Positives", fp)
            c3.metric("False Negatives", fn)
            c4.metric("True Positives", tp)

            st.markdown("""
            **How to interpret the confusion matrix**

            - True Positive: Correctly identified a churned customer.
            - True Negative: Correctly identified a non-churned customer.
            - False Positive: Predicted churn, but the customer did not churn.
            - False Negative: Failed to identify a customer who churned.
            """)

        else:
            st.warning(
                "Could not identify actual and predicted label columns. "
                "Check the prediction CSV column names."
            )


# --------------------------------------------------
# 9. FEATURE IMPORTANCE
# --------------------------------------------------

elif page == "Feature Importance":

    st.title("🔍 Feature Importance Analysis")

    importance_files = list(METRICS_DIR.glob("*importance*.csv"))

    if not importance_files:

        st.warning(
            "Feature-importance CSV files were not found. "
            "Check your model-training output."
        )

    else:

        available_importance = {
            file.stem: file for file in importance_files
        }

        selected_importance = st.selectbox(
            "Select feature importance result",
            list(available_importance.keys())
        )

        importance_data = load_csv(
            available_importance[selected_importance]
        )

        feature_col = find_column(
            importance_data, ["feature", "feature_name"]
        )

        importance_col = find_column(
            importance_data,
            ["importance", "importance_score", "feature_importance"]
        )

        if feature_col and importance_col:

            importance_data = importance_data.sort_values(
                importance_col, ascending=False
            ).head(15)

            fig = px.bar(
                importance_data.sort_values(importance_col),
                x=importance_col,
                y=feature_col,
                orientation="h",
                title="Top 15 Important Features"
            )

            st.plotly_chart(fig, width="stretch")

            st.dataframe(
                importance_data,
                width="stretch"
            )

        else:
            st.warning(
                "The feature-importance CSV does not contain "
                "recognizable feature and importance columns."
            )


# --------------------------------------------------
# 10. CUSTOMER CHURN PREDICTION
# --------------------------------------------------

elif page == "Customer Prediction":

    st.title("👤 Customer Churn Prediction")

    customer_file = find_parquet_file(
        PROCESSED_DIR,
        "customer_features.parquet"
    )

    feature_names_file = MODEL_DIR / "feature_names.json"

    if not customer_file:

        st.warning(
            "customer_features.parquet was not found."
        )

    elif not feature_names_file.exists():

        st.warning(
            "feature_names.json was not found in the models folder."
        )

    elif not models:

        st.warning("No trained models were loaded.")

    else:

        customer_data = load_parquet(customer_file)

        with open(feature_names_file, "r") as f:
            feature_names = json.load(f)

        customer_id_col = find_column(
            customer_data, ["customer_id", "customerid", "id"]
        )

        if customer_id_col is None:

            st.error("Customer ID column was not found.")

        else:

            customer_ids = customer_data[customer_id_col].astype(str)

            selected_customer = st.selectbox(
                "Select a customer",
                customer_ids.tolist()
            )

            selected_model = st.selectbox(
                "Select a prediction model",
                list(models.keys())
            )

            if st.button("Predict Customer Churn"):

                customer_row = customer_data[
                    customer_ids == selected_customer
                ].iloc[[0]]

                missing_features = [
                    feature for feature in feature_names
                    if feature not in customer_row.columns
                ]

                if missing_features:

                    st.error(
                        "Some required model features are missing: "
                        + ", ".join(missing_features[:10])
                    )

                else:

                    X_customer = customer_row[
                        feature_names
                    ].astype(float)

                    model = models[selected_model]

                    prediction = int(model.predict(X_customer)[0])

                    if hasattr(model, "predict_proba"):
                        probability = float(
                            model.predict_proba(X_customer)[0][1]
                        )
                    else:
                        probability = None

                    st.divider()

                    if prediction == 1:

                        st.error("⚠️ Prediction: Customer likely to churn")

                    else:

                        st.success("✅ Prediction: Customer likely to stay")

                    if probability is not None:

                        st.metric(
                            "Predicted Churn Probability",
                            f"{probability * 100:.2f}%"
                        )

                        st.progress(
                            min(max(probability, 0.0), 1.0)
                        )

                    st.caption(
                        "This prediction is generated by the selected "
                        "trained model using the customer's engineered "
                        "telecom usage features."
                    )


# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Ensemble Learning Dashboard | "
    "Sri Lankan Telecom Customer Churn Prediction"
)
