"""Train the custom CNN (Conv32-Conv64-Conv128 -> Dense256 -> Dropout -> Dense128 -> Softmax)."""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import config as C


def build_model(num_classes: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(*C.IMG_SIZE, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    # augmentation (active only in training)
    x = layers.RandomFlip("horizontal_and_vertical")(x)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)

    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        x = layers.MaxPooling2D((2, 2), strides=2)(x)

    x = layers.Flatten()(x)                      # 28*28*128 = 100352
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="chilli_cnn")


def load(path, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        path, image_size=C.IMG_SIZE, batch_size=C.BATCH_SIZE,
        label_mode="categorical", shuffle=shuffle, seed=C.SEED)


def plot_history(h):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h.history["accuracy"], label="train"); ax[0].plot(h.history["val_accuracy"], label="val")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(h.history["loss"], label="train"); ax[1].plot(h.history["val_loss"], label="val")
    ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(C.MODEL_DIR, "training_curves.png"), dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    os.makedirs(C.MODEL_DIR, exist_ok=True)
    train_ds = load(C.TRAIN_DIR, True)
    val_ds = load(C.VAL_DIR, False)
    class_names = train_ds.class_names
    with open(C.CLASSES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)
    print("Classes:", class_names)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model = build_model(len(class_names))
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    cbs = [
        callbacks.ModelCheckpoint(C.MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]
    hist = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=cbs)
    plot_history(hist)
    print(f"Best model saved to {C.MODEL_PATH}")


if __name__ == "__main__":
    main()
