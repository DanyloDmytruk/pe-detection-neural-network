from pathlib import Path

import json
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATASET = Path("data/processed/features.parquet")
RESULTS_DIR = Path("results/baseline")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_parquet(DATASET)

    print(f"Samples: {len(df)}")

    df = df[df["valid_pe"] == True].copy()

    if "parse_error" in df.columns:
        failed = df["parse_error"].notna()
        print(f"Parse failures: {failed.sum()}")

    df = df[~failed].copy()

    print(f"Valid PE samples: {len(df)}")

    print("\nClass distribution:")
    print(df["label"].value_counts())

    # ---------------------------------------------------------
    # Remove columns that must NOT be used as model features
    # ---------------------------------------------------------

    columns_to_remove = [
        "label",
        "filename",
        "path",
        "sha256",
        "parse_error",
        "timestamp"
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in columns_to_remove
    ]

    X = df[feature_columns].copy()
    y = df["label"].astype(int)

    # Convert invalid values to NaN
    X = X.apply(pd.to_numeric, errors="coerce")

    # Replace infinity
    X = X.replace([float("inf"), float("-inf")], pd.NA)

    # LightGBM can handle missing values
    X = X.astype("float32")

    print(f"\nFeatures: {X.shape[1]}")

    # ---------------------------------------------------------
    # Train / validation / test
    # ---------------------------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    print("\nDataset split:")
    print(f"Train:      {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test:       {len(X_test)}")

    # ---------------------------------------------------------
    # LightGBM
    # ---------------------------------------------------------

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    print("\nTraining LightGBM...")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(50),
        ],
    )

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_probability)),
    }

    print("\n==============================")
    print("BASELINE RESULTS")
    print("==============================")

    for name, value in metrics.items():
        print(f"{name:10}: {value:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # ---------------------------------------------------------
    # Save metrics
    # ---------------------------------------------------------

    with open(
        RESULTS_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    # ---------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malware"],
    )

    display.plot()
    plt.title("LightGBM Baseline - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "confusion_matrix.png",
        dpi=200,
    )
    plt.close()

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        RESULTS_DIR / "feature_importance.csv",
        index=False,
    )

    print("\nTop features:")
    print(importance.head(20).to_string(index=False))

    # ---------------------------------------------------------
    # Save model
    # ---------------------------------------------------------

    model.booster_.save_model(
        str(RESULTS_DIR / "model.txt")
    )

    print("\nResults saved to:")
    print(RESULTS_DIR)


if __name__ == "__main__":
    main()