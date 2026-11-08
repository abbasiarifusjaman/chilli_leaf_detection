"""Evaluate every trained model on dataset/test and build a comparison table + chart."""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import config as C
import model_zoo as Z


def main():
    with open(C.CLASSES_PATH) as f:
        class_names = json.load(f)
    ds = tf.keras.utils.image_dataset_from_directory(
        C.TEST_DIR, image_size=C.IMG_SIZE, batch_size=C.BATCH_SIZE,
        label_mode="int", shuffle=False, class_names=class_names)
    y_true = np.concatenate([y.numpy() for _, y in ds])

    rows, preds = [], {}
    for name in Z.MODELS:
        path = C.model_path(name)
        if not os.path.exists(path):
            print(f"skip {name} (not trained)"); continue
        model = tf.keras.models.load_model(path)
        y_pred = np.argmax(model.predict(ds, verbose=0), axis=1)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        rows.append({"model": name, "accuracy": accuracy_score(y_true, y_pred),
                     "precision": p, "recall": r, "f1": f1, "params": model.count_params()})
        preds[name] = y_pred
        tf.keras.backend.clear_session()

    if not rows:
        raise SystemExit("No trained models found. Run: python train_models.py")
    rows.sort(key=lambda d: -d["accuracy"])

    print(f"\n{'Model':18s}{'Acc':>8s}{'Prec':>8s}{'Rec':>8s}{'F1':>8s}{'Params':>14s}")
    for d in rows:
        print(f"{d['model']:18s}{d['accuracy']:8.4f}{d['precision']:8.4f}{d['recall']:8.4f}{d['f1']:8.4f}{d['params']:14,d}")

    with open(os.path.join(C.MODEL_DIR, "comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # bar chart
    x = np.arange(len(rows)); width = 0.2
    plt.figure(figsize=(11, 5))
    for i, k in enumerate(("accuracy", "precision", "recall", "f1")):
        plt.bar(x + (i - 1.5) * width, [d[k] for d in rows], width, label=k)
    plt.xticks(x, [d["model"] for d in rows], rotation=20); plt.ylim(0, 1.05)
    plt.title("Model comparison on test set"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(C.MODEL_DIR, "comparison.png"), dpi=150); plt.close()

    # confusion matrix of the best model
    best = rows[0]["model"]
    cm = confusion_matrix(y_true, preds[best], labels=range(len(class_names)))
    labels = [C.pretty(c) for c in class_names]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion matrix - {best}")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(C.MODEL_DIR, "confusion_matrix_best.png"), dpi=150)
    print(f"\nBest on test set: {best}. Saved comparison.csv / comparison.png / confusion_matrix_best.png in models/")


if __name__ == "__main__":
    main()
