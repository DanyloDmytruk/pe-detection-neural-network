from pathlib import Path
import sys

import lightgbm as lgb
import pandas as pd
import numpy as np
import thrember

from extract_features import extract_pe_features
from extract_api_ngrams import get_imports, make_ngrams


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OWN_MODEL_PATH = BASE_DIR / "results/baseline/model.txt"
EMBER_MODEL_PATH = BASE_DIR / "EMBER2024/results/ember2024/ember_baseline.txt"

API_NGRAMS_PATH = BASE_DIR / "data/processed/api_2grams.csv"


# Number of API 2-grams used by the own model
N_API_NGRAMS = 500


# ============================================================
# Load models
# ============================================================

def load_models():
    own_model = lgb.Booster(
        model_file=str(OWN_MODEL_PATH)
    )

    ember_model = lgb.Booster(
        model_file=str(EMBER_MODEL_PATH)
    )

    return own_model, ember_model


# ============================================================
# Load API n-gram vocabulary
# ============================================================

def load_api_vocabulary():
    df = pd.read_csv(API_NGRAMS_PATH)

    if "api_2gram" not in df.columns:
        raise ValueError(
            f"Column 'api_2gram' not found in {API_NGRAMS_PATH}"
        )

    vocabulary = df["api_2gram"].head(N_API_NGRAMS).tolist()

    if len(vocabulary) != N_API_NGRAMS:
        raise ValueError(
            f"Expected {N_API_NGRAMS} API 2-grams, "
            f"but found {len(vocabulary)}"
        )

    return vocabulary


# ============================================================
# Build own model features
# ============================================================

def build_own_features(exe_path: Path, model):
    # --------------------------------------------------------
    # Base PE features
    # --------------------------------------------------------

    base_features = extract_pe_features(exe_path)

    # --------------------------------------------------------
    # API 2-grams
    # --------------------------------------------------------

    imports = get_imports(exe_path)
    api_2grams = make_ngrams(imports, n=2)

    api_vocabulary = load_api_vocabulary()

    api_features = {}

    api_2gram_set = set(api_2grams)

    for index, ngram in enumerate(api_vocabulary):
        api_features[f"api2_{index}"] = (
            1 if ngram in api_2gram_set else 0
        )

    # --------------------------------------------------------
    # Combine features
    # --------------------------------------------------------

    features = {
        **base_features,
        **api_features,
    }

    # --------------------------------------------------------
    # Use EXACT feature order from LightGBM model
    # --------------------------------------------------------

    feature_names = model.feature_name()

    missing_features = [
        name for name in feature_names
        if name not in features
    ]

    if missing_features:
        raise ValueError(
            "Missing features required by the model:\n"
            + "\n".join(missing_features)
        )

    # Ignore extra features and preserve exact model order
    X = pd.DataFrame(
        [[features[name] for name in feature_names]],
        columns=feature_names,
    )

    # Safety check
    if X.shape[1] != model.num_feature():
        raise ValueError(
            f"Feature count mismatch: "
            f"X has {X.shape[1]}, "
            f"model expects {model.num_feature()}"
        )

    return X


# ============================================================
# EMBER2024 features
# ============================================================

def build_ember_features(exe_path: Path):
    bytez = exe_path.read_bytes()

    extractor = thrember.PEFeatureExtractor()

    vector = extractor.feature_vector(bytez)

    X = np.asarray(vector, dtype=np.float32).reshape(1, -1)

    return X


# ============================================================
# Prediction
# ============================================================

def predict(exe_path: Path):
    if not exe_path.exists():
        raise FileNotFoundError(
            f"File not found: {exe_path}"
        )

    if not exe_path.is_file():
        raise ValueError(
            f"Not a file: {exe_path}"
        )

    print("=" * 60)
    print("Malware Detection")
    print("=" * 60)

    print(f"File: {exe_path}")
    print(f"Size: {exe_path.stat().st_size:,} bytes")
    print()

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    own_model, ember_model = load_models()

    # --------------------------------------------------------
    # Own model
    # --------------------------------------------------------

    X_own = build_own_features(
        exe_path,
        own_model
    )

    own_probability = float(
        own_model.predict(X_own)[0]
    )

    # --------------------------------------------------------
    # EMBER2024 model
    # --------------------------------------------------------

    X_ember = build_ember_features(exe_path)

    if X_ember.shape[1] != ember_model.num_feature():
        raise ValueError(
            f"EMBER feature count mismatch: "
            f"X has {X_ember.shape[1]}, "
            f"model expects {ember_model.num_feature()}"
        )

    ember_probability = float(
        ember_model.predict(X_ember)[0]
    )

    # --------------------------------------------------------
    # Combined probability
    # --------------------------------------------------------

    combined_probability = (
        own_probability + ember_probability
    ) / 2.0

    # --------------------------------------------------------
    # Verdict
    # --------------------------------------------------------

    verdict = (
        "MALWARE"
        if combined_probability >= 0.5
        else "BENIGN"
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    print("Results:")
    print("-" * 60)

    print(
        f"Own model:       "
        f"{own_probability * 100:8.2f}% malware"
    )

    print(
        f"EMBER2024 model: "
        f"{ember_probability * 100:8.2f}% malware"
    )

    print(
        f"Combined:         "
        f"{combined_probability * 100:8.2f}% malware"
    )

    print("-" * 60)

    print(f"Verdict: {verdict}")
    print("=" * 60)


# ============================================================
# CLI
# ============================================================

def main():
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "  python src/predict_exe.py /path/to/file.exe"
        )
        sys.exit(1)

    exe_path = Path(sys.argv[1]).resolve()

    try:
        predict(exe_path)

    except Exception as e:
        print()
        print("ERROR:")
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
