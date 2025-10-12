

import argparse
import json
import os
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------- Helpers --------------------------------- #

def infer_feature_columns(df: pd.DataFrame, target_col: str) -> list:
    """
    Infer feature columns by taking numeric columns and dropping date-like
    and target/revenue columns.
    """
    drop_like = {
        'date', 'year', target_col.lower(),
        'europe_revenue', 'americas_revenue',
        'europe_growth', 'americas_growth'
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    feats = []
    for c in numeric_cols:
        cl = c.lower()
        if any(k in cl for k in drop_like):
            continue
        feats.append(c)

    if len(feats) == 0:
        raise ValueError("No feature columns inferred. Check your CSV.")
    return feats


def binarize_target(series: pd.Series, method: str = "zero") -> pd.Series:
    """
    Convert a continuous growth series into binary labels.
    method options:
      - 'zero':         label=1 if value > 0 else 0
      - 'median':       label=1 if value > median else 0
      - 'percentile:X': label=1 if value >= Xth percentile else 0  (e.g., percentile:60)
    """
    method = method.strip().lower()
    if method == "zero":
        return (series > 0).astype(int)
    elif method == "median":
        thresh = np.nanmedian(series.values)
        return (series > thresh).astype(int)
    elif method.startswith("percentile:"):
        try:
            p = float(method.split(":", 1)[1])
        except Exception:
            raise ValueError("percentile method must be like 'percentile:60'")
        thresh = np.nanpercentile(series.values, p)
        return (series >= thresh).astype(int)
    else:
        raise ValueError("Unknown threshold method. Use 'zero', 'median', or 'percentile:X'.")


def plot_confusion_matrix(cm: np.ndarray, labels=('Low(0)', 'High(1)'), outpath: Path = None):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, interpolation='nearest')
    ax.set_title('Confusion Matrix')
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks); ax.set_xticklabels(labels)
    ax.set_yticks(tick_marks); ax.set_yticklabels(labels)
    ax.set_ylabel('True label'); ax.set_xlabel('Predicted label')

    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_roc(y_true, y_score, outpath: Path = None):
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title('ROC Curve')
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)


# --------------------------- Main train --------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train logistic regression on Chanel growth classification.")
    parser.add_argument("--csv", type=str, default="fred_with_chanel_optionA.csv",
                        help="Path to CSV (default: fred_with_chanel_optionA.csv)")
    parser.add_argument("--target", type=str, default="Europe_Growth",
                        choices=["Europe_Growth", "Americas_Growth"],
                        help="Continuous growth column to binarize and classify.")
    parser.add_argument("--threshold", type=str, default="zero",
                        help="Binarization rule: 'zero', 'median', or 'percentile:X' (e.g., percentile:60)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction (default 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--cv", type=int, default=0,
                        help="Optional StratifiedKFold CV folds on training set (0 disables CV)")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for LogisticRegression")
    parser.add_argument("--penalty", type=str, default="l2", choices=["l2", "none"], help="Penalty type")
    parser.add_argument("--max-iter", type=int, default=2000, help="Max iterations for solver")
    parser.add_argument("--outdir", type=str, default="artifacts", help="Where to save outputs")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(csv_path)

    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not in CSV. Available: {df.columns.tolist()}")

    # Drop rows with missing target and create labels
    y_cont = df[args.target]
    y = binarize_target(y_cont, method=args.threshold)

    # Infer features
    feature_cols = infer_feature_columns(df, target_col=args.target)
    X = df[feature_cols]

    # Build pipeline
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, feature_cols)
        ],
        remainder="drop"
    )

    clf = LogisticRegression(
        penalty=None if args.penalty == "none" else "l2",
        C=args.C,
        max_iter=args.max_iter,
        class_weight="balanced",
        solver="lbfgs" if args.penalty != "none" else "lbfgs",
        n_jobs=None
    )

    pipe = Pipeline(steps=[("preprocess", pre), ("clf", clf)])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Optional cross-validation on train set
    if args.cv and args.cv > 1:
        skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="roc_auc")
        print(f"[CV] ROC AUC (mean±std) across {args.cv} folds: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Fit
    pipe.fit(X_train, y_train)

    # Evaluate
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "target": args.target,
        "threshold_rule": args.threshold,
        "test_size": args.test_size,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }

    print(json.dumps(metrics, indent=2))

    # Save artifacts
    joblib.dump(pipe, outdir / "model.joblib")
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Extract standardized coefficients
    # We do this by fitting the preprocessing separately and then inspecting the classifier coef_
    preprocessor = pipe.named_steps["preprocess"]
    clf_model = pipe.named_steps["clf"]

    # Fit preprocessor on train (already fit in pipeline, but ensure we transform columns deterministically)
    X_train_trans = preprocessor.transform(X_train)
    coef = clf_model.coef_[0]  # shape: (n_features,)

    # Column names after preprocessing (only numeric features here)
    feature_names = feature_cols  # since we only kept numeric

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef_standardized": coef
    }).sort_values("coef_standardized", key=np.abs, ascending=False)

    coef_df.to_csv(outdir / "coefficients.csv", index=False)

    # Plots
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, outpath=outdir / "confusion_matrix.png")
    plot_roc(y_test, y_prob, outpath=outdir / "roc_curve.png")

    print(f"\nArtifacts saved to: {outdir.resolve()}")
    print(" - model.joblib")
    print(" - metrics.json")
    print(" - coefficients.csv")
    print(" - confusion_matrix.png")
    print(" - roc_curve.png")


if __name__ == "__main__":
    main()
