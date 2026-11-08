"""Train all 6 models (or a subset) on dataset/train, validate on dataset/val.

  python train_models.py                       # all six
  python train_models.py --models vgg16 vit    # subset
  python train_models.py --epochs 20

Transfer models: phase 1 = frozen backbone (head only), phase 2 = fine-tune whole network at low LR.
Saves models/<name>.keras, training curves, models/training_summary.json and copies the best
model (by validation accuracy) to models/chilli_cnn.keras so predict.py / app.py use it.
"""
import argparse, json, os, shutil, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks
import config as C
import model_zoo as Z


def load(path, shuffle, bs):
    return tf.keras.utils.image_dataset_from_directory(
        path, image_size=C.IMG_SIZE, batch_size=bs, label_mode="categorical",
        shuffle=shuffle, seed=C.SEED).prefetch(tf.data.AUTOTUNE)


def fit_phase(model, name, train_ds, val_ds, epochs, lr, patience):
    model.compile(optimizer=Z.make_optimizer(name, lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    cbs = [callbacks.EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True),
           callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, patience // 2))]
    return model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs, verbose=2).history


def plot(hist, name):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, k in enumerate(("accuracy", "loss")):
        ax[i].plot(hist[k], label="train"); ax[i].plot(hist["val_" + k], label="val")
        ax[i].set_title(f"{name} - {k}"); ax[i].set_xlabel("epoch"); ax[i].legend()
    fig.tight_layout(); fig.savefig(os.path.join(C.MODEL_DIR, f"curves_{name}.png"), dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(Z.MODELS), choices=list(Z.MODELS))
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--no-promote", action="store_true", help="do not copy best model to chilli_cnn.keras")
    args = ap.parse_args()

    os.makedirs(C.MODEL_DIR, exist_ok=True)
    train_ds = load(C.TRAIN_DIR, True, args.batch_size)
    val_ds = load(C.VAL_DIR, False, args.batch_size)
    class_names = train_ds.element_spec  # placeholder to keep linter quiet
    class_names = tf.keras.utils.image_dataset_from_directory(
        C.TRAIN_DIR, image_size=C.IMG_SIZE, batch_size=1, shuffle=False).class_names
    with open(C.CLASSES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)
    print("Classes:", class_names)

    sum_path = os.path.join(C.MODEL_DIR, "training_summary.json")
    summary = json.load(open(sum_path)) if os.path.exists(sum_path) else {}

    for name in args.models:
        print(f"\n{'=' * 70}\nTraining {name}  ({Z.DESCRIPTIONS[name]})\n{'=' * 70}")
        t0 = time.time()
        model = Z.MODELS[name](len(class_names))
        base = Z.get_base(model)
        hist = {}

        def merge(h):
            for k, v in h.items():
                hist.setdefault(k, []).extend(v)

        if base is not None:                      # transfer learning, two phases
            e1 = max(3, args.epochs // 3)
            merge(fit_phase(model, name, train_ds, val_ds, e1, 1e-3, patience=4))
            base.trainable = True
            merge(fit_phase(model, name, train_ds, val_ds, max(1, args.epochs - e1), 1e-5, patience=6))
        else:                                     # custom CNN / ViT from scratch
            lr = 5e-4 if name == "vit" else 1e-3
            merge(fit_phase(model, name, train_ds, val_ds, args.epochs, lr, patience=8))

        val_loss, val_acc = model.evaluate(val_ds, verbose=0)
        model.save(C.model_path(name))
        plot(hist, name)
        summary[name] = {"val_accuracy": float(val_acc), "val_loss": float(val_loss),
                         "epochs_run": len(hist["accuracy"]), "params": int(model.count_params()),
                         "train_minutes": round((time.time() - t0) / 60, 2)}
        json.dump(summary, open(sum_path, "w"), indent=2)
        print(f"{name}: val_acc={val_acc:.4f}  ({summary[name]['train_minutes']} min)")
        tf.keras.backend.clear_session()

    best = max(summary, key=lambda k: summary[k]["val_accuracy"])
    print("\nValidation accuracy:")
    for k, v in sorted(summary.items(), key=lambda kv: -kv[1]["val_accuracy"]):
        print(f"  {Z.DESCRIPTIONS.get(k, k):32s} {v['val_accuracy']:.4f}")
    if not args.no_promote:
        shutil.copy(C.model_path(best), C.MODEL_PATH)
        print(f"\nBest = {best}  -> copied to {C.MODEL_PATH}")
    print("Next: python compare_models.py")


if __name__ == "__main__":
    main()
