from pathlib import Path

import pandas as pd
import numpy as np


DATASET_PATH = Path("data/processed/features.parquet")


def check_basic_info(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("1. BASIC INFORMATION")
    print("=" * 60)

    print(f"Samples: {len(df)}")
    print(f"Features: {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")


def check_labels(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("2. CLASS DISTRIBUTION")
    print("=" * 60)

    if "label" not in df.columns:
        print("ERROR: 'label' column not found!")
        return

    counts = df["label"].value_counts().sort_index()
    proportions = df["label"].value_counts(normalize=True).sort_index()

    print(counts)
    print("\nProportions:")
    print(proportions.round(4))

    if len(counts) == 2:
        ratio = counts.max() / counts.min()

        print(f"\nClass imbalance ratio: {ratio:.2f}:1")

        if ratio > 1.5:
            print("WARNING: noticeable class imbalance")
        else:
            print("OK: classes are reasonably balanced")


def check_parse_errors(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("3. PARSE ERRORS")
    print("=" * 60)

    if "parse_error" not in df.columns:
        print("No parse_error column.")
        return

    errors = df["parse_error"].notna().sum()

    print(f"Parse errors: {errors}")

    if errors == 0:
        print("OK: no parse errors")
    else:
        print("\nParse errors by class:")

        print(
            df[df["parse_error"].notna()]
            .groupby("label")
            .size()
        )

        print("\nFirst errors:")

        cols = ["filename", "path", "label", "parse_error"]
        available = [c for c in cols if c in df.columns]

        print(
            df[df["parse_error"].notna()][available]
            .head(20)
            .to_string(index=False)
        )


def check_missing_values(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("4. MISSING VALUES")
    print("=" * 60)

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("OK: no missing values")
    else:
        print(missing)


def check_duplicates(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("5. DUPLICATES")
    print("=" * 60)

    if "sha256" in df.columns:
        duplicate_sha = df["sha256"].duplicated(keep=False).sum()

        print(f"Duplicate SHA256 rows: {duplicate_sha}")

        if duplicate_sha == 0:
            print("OK: no duplicate SHA256")
        else:
            print("\nDuplicate files:")

            duplicates = df[df["sha256"].duplicated(keep=False)]

            cols = ["filename", "sha256", "label"]
            available = [c for c in cols if c in duplicates.columns]

            print(
                duplicates[available]
                .sort_values("sha256")
                .to_string(index=False)
            )

    duplicate_filename = df["filename"].duplicated(keep=False).sum()

    print(f"\nDuplicate filenames: {duplicate_filename}")


def check_cross_label_duplicates(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("6. CROSS-LABEL DUPLICATES")
    print("=" * 60)

    if "sha256" not in df.columns:
        print("SHA256 column not found.")
        return

    grouped = df.groupby("sha256")["label"].nunique()

    conflicts = grouped[grouped > 1]

    if conflicts.empty:
        print("OK: no identical files with different labels")
    else:
        print(
            f"WARNING: {len(conflicts)} SHA256 values "
            "have different labels!"
        )

        for sha in conflicts.index:
            rows = df[df["sha256"] == sha]

            print("\n", sha)
            print(
                rows[["filename", "label"]]
                .to_string(index=False)
            )


def check_valid_pe(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("7. PE VALIDITY")
    print("=" * 60)

    if "valid_pe" not in df.columns:
        print("valid_pe column not found.")
        return

    print(df["valid_pe"].value_counts(dropna=False))

    invalid = (df["valid_pe"] == 0).sum()

    print(f"\nInvalid PE: {invalid}")

    if invalid == 0:
        print("OK: all samples are valid PE")
    else:
        print("WARNING: invalid PE files detected")


def check_numeric_features(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("8. NUMERIC FEATURES")
    print("=" * 60)

    excluded = {
        "label",
        "file_size",
        "timestamp",
    }

    numeric = df.select_dtypes(include=np.number).columns

    features = [
        c for c in numeric
        if c not in excluded
    ]

    print(f"Numeric features checked: {len(features)}")

    # Infinite values
    inf_counts = np.isinf(
        df[features].to_numpy()
    ).sum()

    print(f"Infinite values: {inf_counts}")

    # Constant features
    constant = []

    for column in features:
        if df[column].nunique(dropna=False) <= 1:
            constant.append(column)

    if constant:
        print("\nConstant features:")
        for column in constant:
            print(f"  - {column}")
    else:
        print("OK: no constant features")


def check_feature_statistics(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("9. FEATURE STATISTICS")
    print("=" * 60)

    numeric = df.select_dtypes(include=np.number)

    stats = numeric.describe().T

    stats["missing"] = numeric.isna().sum()

    print(
        stats[
            ["count", "mean", "std", "min", "max", "missing"]
        ].to_string()
    )


def check_api_features(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("10. API N-GRAM FEATURES")
    print("=" * 60)

    api_columns = [
        c for c in df.columns
        if c.startswith("api2_")
    ]

    print(f"API 2-gram features: {len(api_columns)}")

    if not api_columns:
        print("No API n-gram features found.")
        return

    zero_rows = (df[api_columns].sum(axis=1) == 0).sum()

    print(
        f"Samples without any selected API 2-gram: "
        f"{zero_rows}"
    )

    print("\nAPI feature density by class:")

    density = df.groupby("label")[api_columns].sum().sum(axis=1)

    print(density)


def check_metadata_leakage(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("11. POTENTIAL DATA LEAKAGE")
    print("=" * 60)

    suspicious = [
        "label",
        "sha256",
        "filename",
        "path",
        "parse_error",
    ]

    print("Metadata columns that should NOT be used for training:")

    for column in suspicious:
        if column in df.columns:
            print(f"  - {column}")

    print(
        "\nMake sure train.py excludes these columns "
        "from X."
    )


def main():
    print("=" * 60)
    print("EMBER2024 DATASET VALIDATION")
    print("=" * 60)

    if not DATASET_PATH.exists():
        print(
            f"\nERROR: Dataset not found:\n"
            f"{DATASET_PATH}"
        )
        return

    print(f"\nLoading: {DATASET_PATH}")

    df = pd.read_parquet(DATASET_PATH)

    check_basic_info(df)
    check_labels(df)
    check_parse_errors(df)
    check_missing_values(df)
    check_duplicates(df)
    check_cross_label_duplicates(df)
    check_valid_pe(df)
    check_numeric_features(df)
    check_feature_statistics(df)
    check_api_features(df)
    check_metadata_leakage(df)

    print("\n" + "=" * 60)
    print("VALIDATION FINISHED")
    print("=" * 60)


if __name__ == "__main__":
    main()