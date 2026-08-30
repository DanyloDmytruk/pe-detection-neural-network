from collections import Counter
from pathlib import Path

import pandas as pd
import pefile


SAMPLES_DIR = Path("data/samples")
FEATURES_PATH = Path("data/processed/features.parquet")


def get_imports(path):
    """
    Extract imported API names from PE file.
    """

    try:
        pe = pefile.PE(path, fast_load=False)

        if not hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
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
    Build n-grams from a sequence.
    """

    if len(items) < n:
        return []

    return [
        "->".join(items[i:i + n])
        for i in range(len(items) - n + 1)
    ]


def main():

    df = pd.read_parquet(FEATURES_PATH)

    counter = Counter()

    processed = 0

    for _, row in df.iterrows():

        path = Path(row["path"])

        imports = get_imports(path)

        if not imports:
            continue

        # API sequence
        ngrams = make_ngrams(
            imports,
            n=2
        )

        counter.update(ngrams)

        processed += 1

    print(f"Processed PE files: {processed}")
    print(f"Unique API 2-grams: {len(counter)}")

    print("\nTop 20 API 2-grams:")

    for ngram, count in counter.most_common(20):
        print(
            f"{ngram}: {count}"
        )

    # Save vocabulary
    output = pd.DataFrame(
        counter.most_common(),
        columns=[
            "api_2gram",
            "frequency",
        ]
    )

    output.to_csv(
        "data/processed/api_2grams.csv",
        index=False
    )

    print(
        "\nSaved: "
        "data/processed/api_2grams.csv"
    )


if __name__ == "__main__":
    main()