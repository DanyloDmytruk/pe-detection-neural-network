from pathlib import Path

import lightgbm as lgb
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


DATASET_PATH = Path("data/processed/features.parquet")
MODEL_PATH = Path("results/baseline/model.txt")
RESULTS_PATH = Path("results/validation/validation_results.txt")


# Службові колонки, які не є ознаками моделі
NON_FEATURE_COLUMNS = {
    "filename",
    "path",
    "sha256",
    "label",
    "parse_error",
}


def load_dataset():
    print("Loading dataset...")

    df = pd.read_parquet(DATASET_PATH)

    print(f"Raw samples: {len(df)}")

    # ---------------------------------------------------------
    # Видаляємо тільки записи з реальною помилкою парсингу.
    # NaN у parse_error НЕ вважаємо помилкою.
    # ---------------------------------------------------------

    if "parse_error" in df.columns:
        parse_errors = df["parse_error"].notna()

        if parse_errors.any():
            print(f"Parse errors found: {parse_errors.sum()}")
            df = df[~parse_errors].copy()

    # ---------------------------------------------------------
    # Залишаємо тільки valid PE.
    #
    # valid_pe у твоєму датасеті має значення 0/1.
    # ---------------------------------------------------------

    if "valid_pe" in df.columns:
        df = df[df["valid_pe"].fillna(0).astype(int) == 1].copy()

    df = df.reset_index(drop=True)

    print(f"Valid PE samples: {len(df)}")

    if len(df) == 0:
        raise ValueError(
            "Dataset is empty after filtering. "
            "Check valid_pe and parse_error columns."
        )

    # ---------------------------------------------------------
    # Завантажуємо модель
    # ---------------------------------------------------------

    print(f"\nLoading model: {MODEL_PATH}")

    model = lgb.Booster(
        model_file=str(MODEL_PATH)
    )

    model_features = model.feature_name()

    print(f"Model features: {len(model_features)}")

    # ---------------------------------------------------------
    # Перевіряємо, що всі ознаки моделі є у датасеті
    # ---------------------------------------------------------

    missing_features = [
        feature
        for feature in model_features
        if feature not in df.columns
    ]

    if missing_features:
        print("\nERROR: Missing features:")

        for feature in missing_features:
            print(f"  {feature}")

        raise ValueError(
            f"Dataset is missing {len(missing_features)} "
            f"features required by the model."
        )

    # ---------------------------------------------------------
    # ВАЖЛИВО:
    #
    # Беремо РІВНО ті 536 колонок,
    # які використовувала модель.
    #
    # Порядок також береться з model.txt.
    # ---------------------------------------------------------

    X = df[model_features].copy()

    y = df["label"].astype(int)

    print(f"Dataset features: {len(df.columns)}")
    print(f"Features used:    {X.shape[1]}")

    print("\nClass distribution:")
    print(y.value_counts())

    return df, X, y, model, model_features


def create_test_split(X, y):
    """
    Формуємо тестову вибірку.

    Модель НЕ перенавчається.
    Ми тільки перевіряємо вже збережену модель.
    """

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        random_state=42,
        stratify=y,
    )

    return X_test, y_test


def evaluate_model(model, X_test, y_test):
    print("\n" + "=" * 50)
    print("MODEL VALIDATION")
    print("=" * 50)

    # ---------------------------------------------------------
    # Отримуємо ймовірність класу 1
    # ---------------------------------------------------------

    probabilities = model.predict(X_test)

    # ---------------------------------------------------------
    # Перетворюємо probability -> class
    # ---------------------------------------------------------

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # ---------------------------------------------------------
    # Метрики
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    print(f"accuracy  : {accuracy:.4f}")
    print(f"precision : {precision:.4f}")
    print(f"recall    : {recall:.4f}")
    print(f"f1        : {f1:.4f}")
    print(f"roc_auc   : {roc_auc:.4f}")

    print("\nClassification report:")

    report = classification_report(
        y_test,
        predictions,
        digits=4,
        zero_division=0,
    )

    print(report)

    # ---------------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    print("Confusion Matrix:")
    print(cm)

    print("\nConfusion Matrix interpretation:")

    print(f"True Negative  (TN): {cm[0, 0]}")
    print(f"False Positive (FP): {cm[0, 1]}")
    print(f"False Negative (FN): {cm[1, 0]}")
    print(f"True Positive  (TP): {cm[1, 1]}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "report": report,
        "confusion_matrix": cm,
    }


def save_results(results):
    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cm = results["confusion_matrix"]

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        file.write("MODEL VALIDATION RESULTS\n")
        file.write("=" * 50)
        file.write("\n\n")

        file.write(
            f"accuracy  : {results['accuracy']:.4f}\n"
        )

        file.write(
            f"precision : {results['precision']:.4f}\n"
        )

        file.write(
            f"recall    : {results['recall']:.4f}\n"
        )

        file.write(
            f"f1        : {results['f1']:.4f}\n"
        )

        file.write(
            f"roc_auc   : {results['roc_auc']:.4f}\n"
        )

        file.write("\n")
        file.write("Classification report:\n")
        file.write(results["report"])

        file.write("\nConfusion Matrix:\n")
        file.write(str(cm))

        file.write("\n\n")
        file.write("Confusion Matrix interpretation:\n")

        file.write(
            f"True Negative  (TN): {cm[0, 0]}\n"
        )

        file.write(
            f"False Positive (FP): {cm[0, 1]}\n"
        )

        file.write(
            f"False Negative (FN): {cm[1, 0]}\n"
        )

        file.write(
            f"True Positive  (TP): {cm[1, 1]}\n"
        )

    print(
        f"\nResults saved to: {RESULTS_PATH}"
    )


def main():

    df, X, y, model, model_features = load_dataset()

    print("\nDataset split:")

    X_test, y_test = create_test_split(
        X,
        y,
    )

    print(f"Total: {len(X)}")
    print(f"Test:  {len(X_test)}")

    print("\nModel:")
    print(f"Path:     {MODEL_PATH}")
    print(f"Features: {len(model_features)}")

    results = evaluate_model(
        model,
        X_test,
        y_test,
    )

    save_results(results)


if __name__ == "__main__":
    main()
