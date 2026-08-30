from pathlib import Path

import pandas as pd
import pefile
from tqdm import tqdm


FEATURES_PATH = Path("data/processed/features.parquet")
NGRAMS_PATH = Path("data/processed/api_2grams.csv")

TOP_K = 500


def get_imports(path):
    """
    Extract imported API names from PE file.
    """

    try:
        pe = pefile.PE(path, fast_load=False)

        if not hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
            pe.close()
            return []

        imports = []

        for dll in pe.DIRECTORY_ENTRY_IMPORT:
            for entry in dll.imports:

                if entry.name is None:
                    continue

                try:
                    name = entry.name.decode(
                        "utf-8",
                        errors="ignore"
                    )
                except Exception:
                    continue

                if name:
                    imports.append(name)

        pe.close()

        return imports

    except Exception:
        return []


def make_ngrams(items, n=2):
    """
    Create API n-grams.
    """

    if len(items) < n:
        return []

    return [
        "->".join(items[i:i + n])
        for i in range(len(items) - n + 1)
    ]


def main():

    print("Loading dataset...")

    df = pd.read_parquet(FEATURES_PATH)

    print(f"Samples: {len(df)}")

    # --------------------------------------------------
    # Load Top-K vocabulary
    # --------------------------------------------------

    ngrams_df = pd.read_csv(NGRAMS_PATH)

    top_ngrams = (
        ngrams_df
        .head(TOP_K)["api_2gram"]
        .tolist()
    )

    print(f"Using Top-{TOP_K} API 2-grams")

    top_ngrams_set = set(top_ngrams)

    # --------------------------------------------------
    # Create columns
    # --------------------------------------------------

    feature_data = []

    print("Extracting API 2-gram features...")

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
    ):

        path = Path(row["path"])

        # All features are initially 0
        sample_features = {
            f"api2_{i}": 0
            for i in range(TOP_K)
        }

        imports = get_imports(path)

        if imports:

            sample_ngrams = make_ngrams(
                imports,
                n=2,
            )

            # Count only vocabulary n-grams
            counts = {}

            for ngram in sample_ngrams:
                if ngram in top_ngrams_set:
                    counts[ngram] = (
                        counts.get(ngram, 0) + 1
                    )

            # Fill features
            for i, ngram in enumerate(top_ngrams):
                sample_features[f"api2_{i}"] = (
                    counts.get(ngram, 0)
                )

        feature_data.append(sample_features)

    # --------------------------------------------------
    # Merge features
    # --------------------------------------------------

    api_features_df = pd.DataFrame(feature_data)

    # Remove old API 2-gram features if script is rerun
    old_columns = [
        column
        for column in df.columns
        if column.startswith("api2_")
    ]

    if old_columns:
        df = df.drop(columns=old_columns)

    df = pd.concat(
        [
            df.reset_index(drop=True),
            api_features_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    df.to_parquet(
        FEATURES_PATH,
        index=False,
    )

    print("\nSaved updated dataset:")
    print(FEATURES_PATH)

    print(f"Total columns: {len(df.columns)}")
    print(f"Added API 2-gram features: {TOP_K}")


if __name__ == "__main__":
    main()