
"""End-to-end starter analysis for Olist delivery performance.

Run from the repository root:
    python src/main.py
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"


def load_data():
    """Load the three required Olist CSV files."""
    required = {
        "orders": "olist_orders_dataset.csv",
        "items": "olist_order_items_dataset.csv",
        "customers": "olist_customers_dataset.csv",
    }
    missing = [filename for filename in required.values()
               if not (RAW_DIR / filename).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset file(s) in data/raw/: "
            + ", ".join(missing)
            + "\nDownload and extract the Olist dataset; see README.md."
        )

    orders = pd.read_csv(RAW_DIR / required["orders"])
    items = pd.read_csv(RAW_DIR / required["items"])
    customers = pd.read_csv(RAW_DIR / required["customers"])
    return orders, items, customers


def prepare_order_level_data(orders, items, customers):
    """Aggregate item rows and join them to one row per order."""
    date_columns = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for column in date_columns:
        orders[column] = pd.to_datetime(orders[column], errors="coerce")

    item_summary = (
        items.groupby("order_id", as_index=False)
        .agg(
            item_count=("order_item_id", "count"),
            total_freight=("freight_value", "sum"),
            total_item_value=("price", "sum"),
            seller_count=("seller_id", "nunique"),
        )
    )

    data = (
        orders.merge(customers, on="customer_id", how="left")
        .merge(item_summary, on="order_id", how="left")
    )

    # Delivery KPIs require both actual and promised delivery timestamps.
    eligible = data.dropna(
        subset=[
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    ).copy()

    eligible["delivery_days"] = (
        eligible["order_delivered_customer_date"]
        - eligible["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    eligible["delay_days"] = (
        eligible["order_delivered_customer_date"]
        - eligible["order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400

    eligible["late"] = (eligible["delay_days"] > 0).astype(int)
    eligible["purchase_month"] = (
        eligible["order_purchase_timestamp"].dt.to_period("M").astype(str)
    )

    # Remove impossible negative lead-time records from KPI/model analysis.
    eligible = eligible.loc[eligible["delivery_days"] >= 0].copy()
    return eligible


def calculate_kpis(data):
    """Return a one-row KPI summary."""
    late_orders = data.loc[data["late"] == 1]
    summary = {
        "eligible_delivered_orders": int(data["order_id"].nunique()),
        "on_time_delivery_rate_pct": round((1 - data["late"].mean()) * 100, 2),
        "late_delivery_rate_pct": round(data["late"].mean() * 100, 2),
        "average_delivery_lead_time_days": round(data["delivery_days"].mean(), 2),
        "median_delivery_lead_time_days": round(data["delivery_days"].median(), 2),
        "average_delay_for_late_orders_days": (
            round(late_orders["delay_days"].mean(), 2)
            if not late_orders.empty else 0.0
        ),
        "average_freight_cost_per_order": round(data["total_freight"].mean(), 2),
    }
    return pd.DataFrame([summary])


def create_segment_tables(data):
    monthly = (
        data.groupby("purchase_month", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            late_rate=("late", "mean"),
            median_delivery_days=("delivery_days", "median"),
        )
    )
    monthly["late_rate_pct"] = (monthly["late_rate"] * 100).round(2)
    monthly = monthly.drop(columns=["late_rate"])

    state = (
        data.groupby("customer_state", as_index=False)
        .agg(
            orders=("order_id", "nunique"),
            late_rate=("late", "mean"),
            median_delivery_days=("delivery_days", "median"),
        )
    )
    state = state.loc[state["orders"] >= 30].copy()
    state["late_rate_pct"] = (state["late_rate"] * 100).round(2)
    state = state.drop(columns=["late_rate"])
    state = state.sort_values("late_rate_pct", ascending=False)
    return monthly, state


def train_baseline_model(data):
    """Train a baseline model with a chronological holdout."""
    numeric_features = [
        "total_freight",
        "total_item_value",
        "item_count",
        "seller_count",
    ]
    categorical_features = ["customer_state"]
    features = numeric_features + categorical_features

    # Exclude rows without a target or essential model fields.
    model_data = data.dropna(subset=["late"]).sort_values(
        "order_purchase_timestamp"
    ).copy()

    if len(model_data) < 100:
        raise ValueError("Not enough eligible rows to create a meaningful train/test split.")

    split_index = int(len(model_data) * 0.8)
    train = model_data.iloc[:split_index]
    test = model_data.iloc[split_index:]

    if train["late"].nunique() < 2:
        raise ValueError("Training data contains only one target class; model cannot be trained.")

    X_train, y_train = train[features], train["late"]
    X_test, y_test = test[features], test["late"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    report = classification_report(
        y_test, predictions, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions)

    metrics = [
        f"Training rows: {len(train)}",
        f"Testing rows: {len(test)}",
        f"Test period: {test['order_purchase_timestamp'].min()} to "
        f"{test['order_purchase_timestamp'].max()}",
        "",
        "Confusion matrix:",
        str(matrix),
        "",
        "Classification report:",
        report,
    ]

    if y_test.nunique() == 2:
        metrics.extend([
            "",
            f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}",
        ])
    else:
        metrics.extend([
            "",
            "ROC-AUC not calculated because the test set contains only one class.",
        ])

    return model, "\n".join(metrics)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading Olist data...")
    orders, items, customers = load_data()

    print("Preparing order-level dataset...")
    data = prepare_order_level_data(orders, items, customers)

    if data.empty:
        raise ValueError("No eligible delivered orders found after data preparation.")

    print("Calculating KPIs...")
    kpis = calculate_kpis(data)
    monthly, state = create_segment_tables(data)

    kpis.to_csv(OUTPUT_DIR / "kpi_summary.csv", index=False)
    monthly.to_csv(OUTPUT_DIR / "monthly_delivery_performance.csv", index=False)
    state.to_csv(OUTPUT_DIR / "state_delivery_performance.csv", index=False)

    print("Training baseline late-delivery model...")
    model, metrics_text = train_baseline_model(data)
    (OUTPUT_DIR / "model_metrics.txt").write_text(metrics_text, encoding="utf-8")
    joblib.dump(model, OUTPUT_DIR / "late_delivery_model.joblib")

    print("\nKPI summary")
    print(kpis.to_string(index=False))
    print(f"\nAnalysis complete. Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
