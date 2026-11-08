"""Evaluate on the test split: Accuracy / Precision / Recall / F1 / Confusion Matrix."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix)
import config as C


def main():
    model = tf.keras.models.load_model(C.MODEL_PATH)
    with open(C.CLASSES_PATH) as f:
        class_names = json.load(f)

    ds = tf.keras.utils.image_dataset_from_directory(
        C.TEST_DIR, image_size=C.IMG_SIZE, batch_size=C.BATCH_SIZE,
        label_mode="int", shuffle=False, class_names=class_names)

    y_true = np.concatenate([y.numpy() for _, y in ds])
    y_pred = np.argmax(model.predict(ds), axis=1)

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    print(f"Accuracy : {acc:.4f}\nPrecision: {p:.4f}\nRecall   : {r:.4f}\nF1-score : {f1:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=[C.pretty(c) for c in class_names], zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=[C.pretty(c) for c in class_names],
                yticklabels=[C.pretty(c) for c in class_names])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    out = os.path.join(C.MODEL_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150)
    print("Saved", out)


if __name__ == "__main__":
    main()
