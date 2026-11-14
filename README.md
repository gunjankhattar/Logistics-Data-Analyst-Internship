
# Logistics Delivery Analytics

A Python project for exploring e-commerce delivery performance and predicting late deliveries.

## Project goal

This project uses the public **Brazilian E-Commerce Public Dataset by Olist** to:
- calculate delivery KPIs;
- explore delay patterns by month and customer state;
- train a baseline late-delivery classification model;
- export summary tables and model evaluation results.

> **Important:** Olist is historical data (2016–2018). It is suitable for learning and demonstrating an analytical workflow, not for representing current carrier performance. This project does not perform real road-route optimisation.

## Repository structure

```text
logistics-delivery-analytics/
├── data/
│   └── raw/                 # Put the downloaded Olist CSV files here (not committed)
├── outputs/                 # Generated reports and metrics (not committed)
├── src/
│   └── main.py              # Data preparation, KPIs, EDA and model
├── requirements.txt
├── .gitignore
└── README.md
```

## 1. Get the dataset

1. Open the [Olist Brazilian E-Commerce dataset on Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
2. Sign in to Kaggle and download the dataset ZIP.
3. Extract the CSV files.
4. Copy the CSV files into `data/raw/`.

The script requires these files:
- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_customers_dataset.csv`

Other Olist files are optional for this first version.

## 2. Set up Python

Use Python 3.10 or newer.

### Windows

Open Command Prompt or the VS Code terminal in this project folder:

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If `py` is unavailable, try `python` instead.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Run the analysis

From the project root folder:

```bash
python src/main.py
```

The script creates:
- `outputs/kpi_summary.csv`
- `outputs/monthly_delivery_performance.csv`
- `outputs/state_delivery_performance.csv`
- `outputs/model_metrics.txt`
- `outputs/late_delivery_model.joblib`

## 4. KPI definitions

- **On-time delivery rate:** delivered orders delivered on or before the estimated date / eligible delivered orders.
- **Late delivery rate:** delivered orders delivered after the estimated date / eligible delivered orders.
- **Average delivery lead time:** average number of days from purchase to actual delivery.
- **Average delay among late orders:** average number of days after the estimated date, for late orders only.
- **Freight cost per order:** average freight value across eligible orders.

Only orders with valid purchase, actual-delivery and estimated-delivery timestamps are included in delivery KPIs. Other order statuses are excluded from this denominator.

## 5. Modelling notes

The baseline model predicts whether a completed order was late. It uses a chronological 80/20 split to approximate prediction on later orders. The current example uses order-level freight, item count, item value, seller count and customer state as predictors.

This is a **retrospective baseline**, not a deployable live alert system. For a real early-warning model, ensure every feature is available at the exact time the prediction is made. In particular, do not use actual delivery dates, review scores or other post-delivery information.

The script reports precision, recall, F1-score and ROC-AUC (where both classes are present). Pay attention to recall for the late class, but also review false positives so operations teams are not overloaded.

## 6. GitHub upload

1. Sign in to [GitHub](https://github.com/).
2. Click **New repository**.
3. Enter a repository name such as `logistics-delivery-analytics`.
4. Choose **Public** or **Private**.
5. Do not add a README or `.gitignore` on GitHub because these are already included in this project.
6. Create the repository.
7. Follow the **uploading an existing project** instructions shown below in the chat response, or use GitHub's web upload option.

Do not upload the raw Kaggle data unless you have checked the dataset's current licence and redistribution terms. This repository's `.gitignore` excludes the data and generated output files by default.
