import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    classification_report
)

# ============================================================
# RESULTS FROM FINAL FEDERATED MODEL
# ============================================================

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# Actual confusion matrix from your test
cm = np.array([
    [186, 124,  59,  31],
    [ 31, 211,  99,  59],
    [  1,   6, 386,   7],
    [  6,  30,   1, 363]
])

# ============================================================
# CREATE RESULTS FOLDER
# ============================================================

import os

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__),
    "results"
)

os.makedirs(
    RESULTS_PATH,
    exist_ok=True
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nGenerating confusion matrix...")

fig, ax = plt.subplots(
    figsize=(8, 7)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

disp.plot(
    ax=ax,
    values_format="d"
)

plt.title(
    "Confusion Matrix - Federated Brain Tumor Model"
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "Actual Class"
)

plt.tight_layout()

confusion_path = os.path.join(
    RESULTS_PATH,
    "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# CALCULATE METRICS
# ============================================================

report = classification_report(
    np.repeat(
        np.arange(4),
        cm.sum(axis=1)
    ),
    np.concatenate([
        np.repeat(0, cm[0, 0]),
        np.repeat(1, cm[0, 1]),
        np.repeat(2, cm[0, 2]),
        np.repeat(3, cm[0, 3]),

        np.repeat(0, cm[1, 0]),
        np.repeat(1, cm[1, 1]),
        np.repeat(2, cm[1, 2]),
        np.repeat(3, cm[1, 3]),

        np.repeat(0, cm[2, 0]),
        np.repeat(1, cm[2, 1]),
        np.repeat(2, cm[2, 2]),
        np.repeat(3, cm[2, 3]),

        np.repeat(0, cm[3, 0]),
        np.repeat(1, cm[3, 1]),
        np.repeat(2, cm[3, 2]),
        np.repeat(3, cm[3, 3])
    ]),
    target_names=CLASS_NAMES,
    output_dict=True
)

precision = [
    report[name]["precision"]
    for name in CLASS_NAMES
]

recall = [
    report[name]["recall"]
    for name in CLASS_NAMES
]

f1 = [
    report[name]["f1-score"]
    for name in CLASS_NAMES
]

# ============================================================
# CLASS-WISE PERFORMANCE
# ============================================================

print("\nGenerating class-wise performance graph...")

x = np.arange(
    len(CLASS_NAMES)
)

width = 0.25

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.bar(
    x - width,
    precision,
    width,
    label="Precision"
)

ax.bar(
    x,
    recall,
    width,
    label="Recall"
)

ax.bar(
    x + width,
    f1,
    width,
    label="F1-Score"
)

ax.set_ylabel(
    "Score"
)

ax.set_xlabel(
    "Brain Tumor Class"
)

ax.set_title(
    "Class-wise Performance of Federated Model"
)

ax.set_xticks(x)

ax.set_xticklabels(
    CLASS_NAMES
)

ax.set_ylim(
    0,
    1
)

ax.legend()

plt.tight_layout()

performance_path = os.path.join(
    RESULTS_PATH,
    "classwise_performance.png"
)

plt.savefig(
    performance_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# FEDERATED LOSS PER ROUND
# ============================================================

rounds = [
    1,
    2,
    3,
    4
]

losses = [
    1.0158832669,
    0.6263765693,
    0.5108583793,
    0.4646798447
]

print("\nGenerating federated loss graph...")

fig, ax = plt.subplots(
    figsize=(8, 6)
)

ax.plot(
    rounds,
    losses,
    marker="o"
)

ax.set_xlabel(
    "Federated Round"
)

ax.set_ylabel(
    "Global Loss"
)

ax.set_title(
    "Federated Learning Global Loss Across Rounds"
)

ax.set_xticks(rounds)

ax.grid(
    True
)

plt.tight_layout()

loss_path = os.path.join(
    RESULTS_PATH,
    "federated_loss.png"
)

plt.savefig(
    loss_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n==========================================")
print("FEDERATED MODEL RESULTS")
print("==========================================")

print(
    "Test Accuracy : 71.63%"
)

print(
    "Test Loss     : 0.8770"
)

print(
    "Macro F1      : 70.12%"
)

print(
    "Macro Precision : 72.41%"
)

print(
    "Macro Recall    : 71.63%"
)

print("\nResults saved in:")

print(
    RESULTS_PATH
)

print("\nGenerated files:")

print(
    "1. confusion_matrix.png"
)

print(
    "2. classwise_performance.png"
)

print(
    "3. federated_loss.png"
)

print("\n==========================================")
print("DONE")
print("==========================================")