import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


RESULTS_DIR = "results"
OUTPUT_DIR = "results/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# RESULTS
# ============================================================

baseline = {
    "accuracy": 0.9530,
    "precision": 0.9722,
    "recall": 0.9333,
    "f1": 0.9524,
    "roc_auc": 0.9950,
}

api_ngrams = {
    "accuracy": 0.9664,
    "precision": 0.9861,
    "recall": 0.9467,
    "f1": 0.9660,
    "roc_auc": 0.9960,
}

validation = {
    "accuracy": 0.9933,
    "precision": 1.0000,
    "recall": 0.9867,
    "f1": 0.9933,
    "roc_auc": 0.9991,
}

external_cm = np.array([
    [343, 13],
    [4, 356]
])


# Calculate external metrics
tn, fp, fn, tp = external_cm.ravel()

external_accuracy = (tp + tn) / (tp + tn + fp + fn)
external_precision = tp / (tp + fp)
external_recall = tp / (tp + fn)
external_f1 = (
    2 * external_precision * external_recall
    / (external_precision + external_recall)
)

external = {
    "accuracy": external_accuracy,
    "precision": external_precision,
    "recall": external_recall,
    "f1": external_f1,
    "roc_auc": np.nan,
}


# ============================================================
# DATAFRAME
# ============================================================

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]

comparison = pd.DataFrame(
    {
        "Baseline (36)": baseline,
        "API 2-grams (536)": api_ngrams,
        "Validation (149)": validation,
        "External (360)": external,
    }
).T

comparison.columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
]


# ============================================================
# SAVE CSV
# ============================================================

comparison.to_csv(
    f"{OUTPUT_DIR}/comparison_table.csv",
    float_format="%.4f"
)


# ============================================================
# PRINT TABLE
# ============================================================

print("\n" + "=" * 80)
print("MODEL RESULTS COMPARISON")
print("=" * 80)

print(
    comparison.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# GRAPH 1 — METRICS COMPARISON
# ============================================================

plt.figure(figsize=(12, 7))

x = np.arange(len(metrics))
width = 0.2

models = [
    ("Baseline (36)", baseline),
    ("API 2-grams (536)", api_ngrams),
    ("Validation (149)", validation),
    ("External (360)", external),
]

for i, (name, data) in enumerate(models):
    values = [
        data[m] if not np.isnan(data[m]) else 0
        for m in metrics
    ]

    plt.bar(
        x + (i - 1.5) * width,
        values,
        width,
        label=name
    )

plt.xticks(
    x,
    ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
)

plt.ylim(0.85, 1.01)
plt.ylabel("Score")
plt.title("Comparison of Malware Detection Model Performance")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/metrics_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 2 — BASELINE VS API N-GRAMS
# ============================================================

plt.figure(figsize=(10, 6))

x = np.arange(len(metrics))
width = 0.35

baseline_values = [
    baseline[m] for m in metrics
]

api_values = [
    api_ngrams[m] for m in metrics
]

plt.bar(
    x - width / 2,
    baseline_values,
    width,
    label="Baseline"
)

plt.bar(
    x + width / 2,
    api_values,
    width,
    label="Baseline + API 2-grams"
)

plt.xticks(
    x,
    ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
)

plt.ylim(0.90, 1.01)
plt.ylabel("Score")
plt.title("Effect of API 2-gram Features")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/api_ngram_improvement.png",
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 3 — CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=external_cm,
    display_labels=["Benign", "Malware"]
)

disp.plot(
    ax=ax,
    values_format="d",
    colorbar=False
)

plt.title(
    "Confusion Matrix — External Dataset (360 Samples)"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/external_confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 4 — FEATURE INCREASE
# ============================================================

plt.figure(figsize=(8, 6))

features = [36, 536]

plt.bar(
    ["Baseline", "API 2-grams"]
    ,
    features
)

plt.ylabel("Number of features")
plt.title("Feature Space Expansion")

for i, value in enumerate(features):
    plt.text(
        i,
        value + 10,
        str(value),
        ha="center"
    )

plt.ylim(0, 600)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/feature_increase.png",
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 5 — ERROR DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

errors = {
    "False Positives": fp,
    "False Negatives": fn,
}

plt.bar(
    errors.keys(),
    errors.values()
)

plt.ylabel("Number of samples")
plt.title("Classification Errors — External Dataset")

for i, value in enumerate(errors.values()):
    plt.text(
        i,
        value + 0.5,
        str(value),
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/external_errors.png",
    dpi=300
)

plt.close()


# ============================================================
# SAVE TEXT REPORT
# ============================================================

with open(
    f"{OUTPUT_DIR}/comparison_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("MODEL RESULTS COMPARISON\n")
    f.write("=" * 80 + "\n\n")

    f.write(
        comparison.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    f.write("\n\n")

    f.write("EXTERNAL CONFUSION MATRIX\n")
    f.write("=" * 80 + "\n")

    f.write(f"TN: {tn}\n")
    f.write(f"FP: {fp}\n")
    f.write(f"FN: {fn}\n")
    f.write(f"TP: {tp}\n\n")

    f.write("External metrics:\n")
    f.write(f"Accuracy:  {external_accuracy:.4f}\n")
    f.write(f"Precision: {external_precision:.4f}\n")
    f.write(f"Recall:    {external_recall:.4f}\n")
    f.write(f"F1:        {external_f1:.4f}\n")


print("\n" + "=" * 80)
print("FILES CREATED")
print("=" * 80)

print(f"{OUTPUT_DIR}/metrics_comparison.png")
print(f"{OUTPUT_DIR}/api_ngram_improvement.png")
print(f"{OUTPUT_DIR}/external_confusion_matrix.png")
print(f"{OUTPUT_DIR}/feature_increase.png")
print(f"{OUTPUT_DIR}/external_errors.png")
print(f"{OUTPUT_DIR}/comparison_table.csv")
print(f"{OUTPUT_DIR}/comparison_report.txt")