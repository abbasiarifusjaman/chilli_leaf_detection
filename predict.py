"""Predict disease for a single leaf image and print treatment advice."""
import argparse, json
import numpy as np
import tensorflow as tf
from PIL import Image
import config as C


def load_model_and_classes():
    model = tf.keras.models.load_model(C.MODEL_PATH)
    with open(C.CLASSES_PATH) as f:
        return model, json.load(f)


def predict_image(model, class_names, img: Image.Image, top_k=3):
    img = img.convert("RGB").resize(C.IMG_SIZE)
    arr = np.expand_dims(np.asarray(img, dtype="float32"), 0)  # model rescales internally
    probs = model.predict(arr, verbose=0)[0]
    idx = np.argsort(probs)[::-1][:top_k]
    return [(class_names[i], float(probs[i])) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    args = ap.parse_args()
    model, classes = load_model_and_classes()
    results = predict_image(model, classes, Image.open(args.image))
    best, conf = results[0]
    print(f"\nPrediction : {C.pretty(best)}  ({conf*100:.1f}%)")
    print("Top-3      :", ", ".join(f"{C.pretty(c)} {p*100:.1f}%" for c, p in results))
    print("Treatment  :", C.TREATMENTS.get(best, "Consult a local agricultural expert."))


if __name__ == "__main__":
    main()
