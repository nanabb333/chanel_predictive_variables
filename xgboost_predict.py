
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor, plot_importance
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ------------------ Config ------------------ #
DATA_PATH = "dataset.csv"
OUTDIR = Path("xgboost_outputs")
OUTDIR.mkdir(exist_ok=True)

TARGETS = {
    "Europe": "Europe_Revenue",
    "Americas": "Americas_Revenue"
}

# ------------------ Helpers ------------------ #
def select_features_by_prefix(df: pd.DataFrame, region: str) -> pd.DataFrame:
    prefix = "US_" if region == "Americas" else "EU_"
    feat_cols = [c for c in df.columns if c.startswith(prefix)]
    if not feat_cols:
        raise ValueError(f"No features found with prefix '{prefix}'.")
    X = df[feat_cols].copy()
    X = X.fillna(method="ffill").fillna(method="bfill")
    return X


def evaluate_model(y_true, y_pred, y_true_log, y_pred_log):
    """Compute metrics on both original and log scale."""
    metrics = {
        "r2_original": float(r2_score(y_true, y_pred)),
        "rmse_original": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae_original": float(mean_absolute_error(y_true, y_pred)),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
    }
    return metrics


def plot_feature_importance(model, region):
    plt.figure(figsize=(8, 6))
    plot_importance(model, max_num_features=10, importance_type="weight")
    plt.title(f"{region} — Top Feature Importance")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"feature_importance_{region.lower()}.png", dpi=200)
    plt.close()


def plot_predicted_vs_actual(y_test, y_pred, region):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Revenue")
    plt.ylabel("Predicted Revenue")
    plt.title(f"{region} — XGBoost Prediction vs Actual")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"predicted_vs_actual_{region.lower()}.png", dpi=200)
    plt.close()


# ------------------ Training ------------------ #
def train_region(df, region):
    target_col = TARGETS[region]
    y_raw = df[target_col].copy()
    X = select_features_by_prefix(df, region)

    # log-transform target
    y_log = np.log1p(y_raw)

    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="rmse"
    )

    model.fit(X_train, y_train_log)

    # Predict on log scale and invert back
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test = np.expm1(y_test_log)

    # Evaluate
    metrics = evaluate_model(y_test, y_pred, y_test_log, y_pred_log)
    print(f"\n📈 {region} Model Performance:")
    print(json.dumps(metrics, indent=2))

    # Save metrics
    with open(OUTDIR / f"metrics_{region.lower()}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save feature importances
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    importance.to_csv(OUTDIR / f"feature_importance_{region.lower()}.csv", index=False)

    # Plots
    plot_feature_importance(model, region)
    plot_predicted_vs_actual(y_test, y_pred, region)


# ------------------ Main ------------------ #
def main():
    df = pd.read_csv(DATA_PATH)
    for region in TARGETS.keys():
        train_region(df, region)
    print("\n✅ All models completed. Results saved in 'xgboost_outputs/' folder.")


if __name__ == "__main__":
    main()
