from pathlib import Path
import hashlib
import math
import re

import numpy as np
import pandas as pd
import pefile
import lightgbm as lgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

BENIGN_DIR = Path("data/external_test/benign")
MALWARE_DIR = Path("data/external_test/malware")

MODEL_PATH = Path("results/baseline/model.txt")

RESULTS_DIR = Path("results/external_test")
RESULTS_CSV = RESULTS_DIR / "predictions.csv"
RESULTS_TXT = RESULTS_DIR / "results.txt"


# ============================================================
# CONSTANTS
# ============================================================

API_NGRAM_FILE = Path(
    "data/processed/api_2grams.csv"
)

TOP_N_API_NGRAMS = 500


# ============================================================
# BASIC UTILITIES
# ============================================================

def calculate_entropy(data: bytes) -> float:
    """
    Calculate Shannon entropy.
    """

    if not data:
        return 0.0

    counts = np.bincount(
        np.frombuffer(data, dtype=np.uint8),
        minlength=256,
    )

    probabilities = counts[counts > 0] / len(data)

    return float(
        -np.sum(
            probabilities * np.log2(probabilities)
        )
    )


def calculate_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# API 2-GRAM EXTRACTION
# ============================================================

def load_top_api_ngrams():
    """
    Load the same Top-500 API 2-grams that were used
    during training.
    """

    if not API_NGRAM_FILE.exists():
        raise FileNotFoundError(
            f"API n-gram file not found: {API_NGRAM_FILE}"
        )

    df = pd.read_csv(API_NGRAM_FILE)

    # The original extractor generated columns
    # based on the most frequent 2-grams.

    # Try to detect the n-gram column automatically.
    possible_columns = [
        "ngram",
        "api_2gram",
        "api2gram",
        "gram",
        "pair",
    ]

    ngram_column = None

    for column in possible_columns:
        if column in df.columns:
            ngram_column = column
            break

    if ngram_column is None:
        # If the first column contains the n-gram strings,
        # use it.
        for column in df.columns:
            if df[column].dtype == "object":
                ngram_column = column
                break

    if ngram_column is None:
        raise ValueError(
            "Could not determine API 2-gram column "
            f"in {API_NGRAM_FILE}"
        )

    top_ngrams = (
        df[ngram_column]
        .dropna()
        .astype(str)
        .head(TOP_N_API_NGRAMS)
        .tolist()
    )

    print(
        f"Loaded API 2-grams: {len(top_ngrams)}"
    )

    return top_ngrams


def normalize_api_name(name: str) -> str:
    """
    Normalize imported API name.
    """

    return name.strip()


def extract_api_ngrams(pe: pefile.PE):
    """
    Extract API 2-grams from imported functions.

    Returns:
        set of strings:
        API1->API2
    """

    api_names = []

    if not hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        return set()

    for entry in pe.DIRECTORY_ENTRY_IMPORT:

        for imp in entry.imports:

            if imp.name is None:
                continue

            try:
                name = imp.name.decode(
                    "utf-8",
                    errors="ignore",
                )
            except Exception:
                continue

            name = normalize_api_name(name)

            if name:
                api_names.append(name)

    # Remove duplicates while preserving order
    api_names = list(
        dict.fromkeys(api_names)
    )

    ngrams = set()

    for i in range(len(api_names) - 1):
        ngram = (
            f"{api_names[i]}->{api_names[i + 1]}"
        )

        ngrams.add(ngram)

    return ngrams


# ============================================================
# PE FEATURE EXTRACTION
# ============================================================

def extract_pe_features(path: Path):
    """
    Extract the same general PE features used by the model.

    Returns dictionary.
    """

    features = {
        "filename": path.name,
        "path": str(path),
        "file_size": path.stat().st_size,
        "sha256": calculate_sha256(path),

        "valid_pe": 0,
        "machine": 0,
        "number_of_sections_header": 0,
        "timestamp": 0,
        "characteristics": 0,
        "entry_point": 0,
        "image_base": 0,
        "subsystem": 0,
        "section_count": 0,

        "section_entropy_min": 0.0,
        "section_entropy_max": 0.0,
        "section_entropy_mean": 0.0,
        "section_entropy_std": 0.0,

        "executable_sections": 0,
        "writable_sections": 0,
        "readable_sections": 0,
        "rwx_sections": 0,

        "section_raw_size_min": 0,
        "section_raw_size_max": 0,
        "section_raw_size_mean": 0.0,

        "section_virtual_size_min": 0,
        "section_virtual_size_max": 0,
        "section_virtual_size_mean": 0.0,

        "section_size_ratio_min": 0.0,
        "section_size_ratio_max": 0.0,
        "section_size_ratio_mean": 0.0,

        "high_entropy_sections": 0,

        "imported_dlls": 0,
        "imported_functions": 0,

        "file_operations": 0,
        "registry_operations": 0,
        "network_operations": 0,
        "process_operations": 0,
        "memory_operations": 0,
        "crypto_operations": 0,
        "service_operations": 0,
    }

    try:
        pe = pefile.PE(
            str(path),
            fast_load=False,
        )

        features["valid_pe"] = 1

        # ----------------------------------------------------
        # PE HEADER
        # ----------------------------------------------------

        features["machine"] = int(
            pe.FILE_HEADER.Machine
        )

        features[
            "number_of_sections_header"
        ] = int(
            pe.FILE_HEADER.NumberOfSections
        )

        features["timestamp"] = int(
            pe.FILE_HEADER.TimeDateStamp
        )

        features["characteristics"] = int(
            pe.FILE_HEADER.Characteristics
        )

        features["entry_point"] = int(
            pe.OPTIONAL_HEADER.AddressOfEntryPoint
        )

        features["image_base"] = int(
            pe.OPTIONAL_HEADER.ImageBase
        )

        features["subsystem"] = int(
            pe.OPTIONAL_HEADER.Subsystem
        )

        # ----------------------------------------------------
        # SECTIONS
        # ----------------------------------------------------

        sections = pe.sections

        features["section_count"] = len(
            sections
        )

        entropies = []
        raw_sizes = []
        virtual_sizes = []
        size_ratios = []

        executable = 0
        writable = 0
        readable = 0
        rwx = 0
        high_entropy = 0

        for section in sections:

            try:
                data = section.get_data()
            except Exception:
                data = b""

            entropy = calculate_entropy(data)

            entropies.append(entropy)

            if entropy >= 7.0:
                high_entropy += 1

            raw_size = int(
                section.SizeOfRawData
            )

            virtual_size = int(
                section.Misc_VirtualSize
            )

            raw_sizes.append(raw_size)
            virtual_sizes.append(virtual_size)

            if virtual_size > 0:
                ratio = (
                    raw_size / virtual_size
                )
            else:
                ratio = 0.0

            size_ratios.append(ratio)

            characteristics = int(
                section.Characteristics
            )

            # IMAGE_SCN_MEM_EXECUTE
            is_executable = bool(
                characteristics & 0x20000000
            )

            # IMAGE_SCN_MEM_WRITE
            is_writable = bool(
                characteristics & 0x80000000
            )

            # IMAGE_SCN_MEM_READ
            is_readable = bool(
                characteristics & 0x40000000
            )

            if is_executable:
                executable += 1

            if is_writable:
                writable += 1

            if is_readable:
                readable += 1

            if (
                is_executable
                and is_writable
                and is_readable
            ):
                rwx += 1

        if entropies:
            features[
                "section_entropy_min"
            ] = min(entropies)

            features[
                "section_entropy_max"
            ] = max(entropies)

            features[
                "section_entropy_mean"
            ] = float(np.mean(entropies))

            features[
                "section_entropy_std"
            ] = float(np.std(entropies))

        features[
            "executable_sections"
        ] = executable

        features[
            "writable_sections"
        ] = writable

        features[
            "readable_sections"
        ] = readable

        features[
            "rwx_sections"
        ] = rwx

        features[
            "high_entropy_sections"
        ] = high_entropy

        if raw_sizes:
            features[
                "section_raw_size_min"
            ] = min(raw_sizes)

            features[
                "section_raw_size_max"
            ] = max(raw_sizes)

            features[
                "section_raw_size_mean"
            ] = float(np.mean(raw_sizes))

        if virtual_sizes:
            features[
                "section_virtual_size_min"
            ] = min(virtual_sizes)

            features[
                "section_virtual_size_max"
            ] = max(virtual_sizes)

            features[
                "section_virtual_size_mean"
            ] = float(np.mean(virtual_sizes))

        if size_ratios:
            features[
                "section_size_ratio_min"
            ] = min(size_ratios)

            features[
                "section_size_ratio_max"
            ] = max(size_ratios)

            features[
                "section_size_ratio_mean"
            ] = float(np.mean(size_ratios))

        # ----------------------------------------------------
        # IMPORTS
        # ----------------------------------------------------

        imported_dlls = 0
        imported_functions = 0

        if hasattr(
            pe,
            "DIRECTORY_ENTRY_IMPORT",
        ):
            imported_dlls = len(
                pe.DIRECTORY_ENTRY_IMPORT
            )

            for entry in (
                pe.DIRECTORY_ENTRY_IMPORT
            ):
                imported_functions += len(
                    entry.imports
                )

        features[
            "imported_dlls"
        ] = imported_dlls

        features[
            "imported_functions"
        ] = imported_functions

        # ----------------------------------------------------
        # API 2-GRAMS
        # ----------------------------------------------------

        api_ngrams = extract_api_ngrams(pe)

        features["_api_ngrams"] = api_ngrams

        pe.close()

        return features

    except Exception as exc:

        print(
            f"[ERROR] {path.name}: {exc}"
        )

        features["_api_ngrams"] = set()

        return features


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_api_features(
    feature_dict,
    top_ngrams,
):
    """
    Add api2_0 ... api2_499 features.
    """

    api_ngrams = feature_dict.get(
        "_api_ngrams",
        set(),
    )

    for index, ngram in enumerate(
        top_ngrams
    ):
        feature_dict[
            f"api2_{index}"
        ] = int(
            ngram in api_ngrams
        )

    return feature_dict


# ============================================================
# COLLECT FILES
# ============================================================

def collect_files():
    files = []

    for directory, label in [
        (BENIGN_DIR, 0),
        (MALWARE_DIR, 1),
    ]:

        if not directory.exists():
            print(
                f"WARNING: Directory does not exist: "
                f"{directory}"
            )

            continue

        for path in directory.rglob("*"):

            if not path.is_file():
                continue

            # External dataset should contain PE files.
            # We allow all files here and let pefile validate them.
            files.append(
                (path, label)
            )

    return files


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    print(
        f"\nLoading model: {MODEL_PATH}"
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = lgb.Booster(
        model_file=str(MODEL_PATH)
    )

    model_features = model.feature_name()

    print(
        f"Model expects "
        f"{len(model_features)} features"
    )

    return model, model_features


# ============================================================
# MAIN EXTRACTION
# ============================================================

def build_external_dataset(
    files,
    top_ngrams,
):
    rows = []

    print(
        f"\nProcessing external samples: "
        f"{len(files)}"
    )

    for index, (path, label) in enumerate(
        files,
        start=1,
    ):

        print(
            f"[{index}/{len(files)}] "
            f"{path.name}"
        )

        features = extract_pe_features(
            path
        )

        # Skip invalid PE files
        if features["valid_pe"] != 1:
            print(
                "  -> skipped: invalid PE"
            )
            continue

        features = add_api_features(
            features,
            top_ngrams,
        )

        features["label"] = label

        # Internal helper is not a model feature.
        features.pop(
            "_api_ngrams",
            None,
        )

        rows.append(features)

    if not rows:
        raise ValueError(
            "No valid PE samples found "
            "in external_test."
        )

    df = pd.DataFrame(rows)

    return df


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    df,
    model_features,
):
    print(
        "\n" + "=" * 50
    )

    print(
        "EXTERNAL TEST RESULTS"
    )

    print(
        "=" * 50
    )

    # --------------------------------------------------------
    # Verify required features
    # --------------------------------------------------------

    missing = [
        feature
        for feature in model_features
        if feature not in df.columns
    ]

    if missing:

        print(
            "\nMissing model features:"
        )

        for feature in missing:
            print(
                f"  {feature}"
            )

        raise ValueError(
            f"Missing {len(missing)} model features."
        )

    # --------------------------------------------------------
    # EXACT feature order from model
    # --------------------------------------------------------

    X = df[
        model_features
    ].copy()

    y = df["label"].astype(int)

    print(
        f"Samples:  {len(df)}"
    )

    print(
        f"Features: {len(model_features)}"
    )

    print(
        "\nClass distribution:"
    )

    print(
        y.value_counts()
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probabilities = model.predict(X)

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y,
        predictions,
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    # ROC-AUC requires both classes
    if len(
        np.unique(y)
    ) == 2:

        roc_auc = roc_auc_score(
            y,
            probabilities,
        )

    else:

        roc_auc = float("nan")

    print(
        f"\nAccuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1        : {f1:.4f}"
    )

    if math.isnan(roc_auc):

        print(
            "ROC-AUC   : N/A"
        )

    else:

        print(
            f"ROC-AUC   : {roc_auc:.4f}"
        )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report = classification_report(
        y,
        predictions,
        digits=4,
        zero_division=0,
    )

    print(
        "\nClassification report:"
    )

    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y,
        predictions,
    )

    print(
        "Confusion Matrix:"
    )

    print(cm)

    print(
        "\nConfusion Matrix interpretation:"
    )

    print(
        f"True Negative  (TN): {cm[0, 0]}"
    )

    print(
        f"False Positive (FP): {cm[0, 1]}"
    )

    print(
        f"False Negative (FN): {cm[1, 0]}"
    )

    print(
        f"True Positive  (TP): {cm[1, 1]}"
    )

    # --------------------------------------------------------
    # Save per-file predictions
    # --------------------------------------------------------

    result_df = df[
        [
            "filename",
            "path",
            "sha256",
            "label",
        ]
    ].copy()

    result_df[
        "probability_malware"
    ] = probabilities

    result_df[
        "prediction"
    ] = predictions

    result_df[
        "correct"
    ] = (
        result_df["label"]
        == result_df["prediction"]
    )

    result_df[
        "result"
    ] = np.where(
        result_df["prediction"] == 1,
        "MALWARE",
        "BENIGN",
    )

    result_df.to_csv(
        RESULTS_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Save textual results
    # --------------------------------------------------------

    with open(
        RESULTS_TXT,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "EXTERNAL TEST RESULTS\n"
        )

        file.write(
            "=" * 50
            + "\n\n"
        )

        file.write(
            f"Samples: {len(df)}\n\n"
        )

        file.write(
            f"Accuracy:  {accuracy:.4f}\n"
        )

        file.write(
            f"Precision: {precision:.4f}\n"
        )

        file.write(
            f"Recall:    {recall:.4f}\n"
        )

        file.write(
            f"F1:        {f1:.4f}\n"
        )

        if math.isnan(roc_auc):

            file.write(
                "ROC-AUC:   N/A\n"
            )

        else:

            file.write(
                f"ROC-AUC:   {roc_auc:.4f}\n"
            )

        file.write(
            "\nClassification report:\n"
        )

        file.write(report)

        file.write(
            "\nConfusion Matrix:\n"
        )

        file.write(
            str(cm)
        )

        file.write(
            "\n\nConfusion Matrix interpretation:\n"
        )

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
        f"\nPredictions saved to:"
        f"\n{RESULTS_CSV}"
    )

    print(
        f"\nResults saved to:"
        f"\n{RESULTS_TXT}"
    )

    return result_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 50)
    print("EXTERNAL MODEL TEST")
    print("=" * 50)

    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    if not BENIGN_DIR.exists():
        raise FileNotFoundError(
            f"Missing directory: {BENIGN_DIR}"
        )

    if not MALWARE_DIR.exists():
        raise FileNotFoundError(
            f"Missing directory: {MALWARE_DIR}"
        )

    # --------------------------------------------------------
    # Load API vocabulary
    # --------------------------------------------------------

    top_ngrams = load_top_api_ngrams()

    # --------------------------------------------------------
    # Collect external samples
    # --------------------------------------------------------

    files = collect_files()

    if not files:
        raise ValueError(
            "No files found in external_test."
        )

    print(
        f"\nFound external files: "
        f"{len(files)}"
    )

    benign_count = sum(
        label == 0
        for _, label in files
    )

    malware_count = sum(
        label == 1
        for _, label in files
    )

    print(
        f"Benign:  {benign_count}"
    )

    print(
        f"Malware: {malware_count}"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, model_features = load_model()

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    df = build_external_dataset(
        files,
        top_ngrams,
    )

    print(
        f"\nValid external PE samples: "
        f"{len(df)}"
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluate(
        model,
        df,
        model_features,
    )


if __name__ == "__main__":
    main()