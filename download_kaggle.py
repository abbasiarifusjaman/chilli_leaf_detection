"""Download a chilli leaf dataset from Kaggle and sort it into dataset/all/<class>/.

Setup: pip install kaggle, then put kaggle.json in ~/.kaggle/ (Kaggle > Account > Create API Token).
Usage: python download_kaggle.py --dataset <owner/dataset-slug>
Find a chilli/pepper leaf disease dataset on kaggle.com and pass its slug.
"""
import argparse, os, shutil, subprocess, sys
import config as C

RAW = os.path.join(C.BASE_DIR, "kaggle_raw")
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# keyword in raw folder name -> our class folder (first match wins)
KEYWORDS = [
    ("healthy", "healthy"), ("bacterial", "bacterial_leaf_spot"), ("cercospora", "cercospora_leaf_spot"),
    ("frog", "cercospora_leaf_spot"), ("anthracnose", "anthracnose"), ("powdery", "powdery_mildew"),
    ("downy", "downy_mildew"), ("mosaic", "mosaic_virus"), ("curl", "leaf_curl_virus"),
    ("blight", "leaf_blight"), ("nutrient", "nutrient_deficiency"), ("deficien", "nutrient_deficiency"),
    ("yellow", "nutrient_deficiency"),
]


def map_class(folder):
    f = folder.lower()
    for key, cls in KEYWORDS:
        if key in f:
            return cls
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g. owner/chilli-leaf-disease")
    args = ap.parse_args()

    os.makedirs(RAW, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", args.dataset, "-p", RAW, "--unzip"]
    print(" ".join(cmd))
    if subprocess.call(cmd) != 0:
        sys.exit("Kaggle download failed. Check kaggle.json and the dataset slug.")

    copied, skipped = {}, set()
    for root, _, files in os.walk(RAW):
        imgs = [f for f in files if f.lower().endswith(EXT)]
        if not imgs:
            continue
        cls = map_class(os.path.basename(root))
        if cls is None:
            skipped.add(os.path.basename(root)); continue
        dest = os.path.join(C.ALL_DIR, cls)
        os.makedirs(dest, exist_ok=True)
        for f in imgs:
            shutil.copy2(os.path.join(root, f), os.path.join(dest, f"{copied.get(cls, 0):05d}_{f}"))
            copied[cls] = copied.get(cls, 0) + 1
    print("Copied:", copied)
    if skipped:
        print("Unmapped folders (rename manually into dataset/all/<class>):", sorted(skipped))
    print("Next: python split_dataset.py")


if __name__ == "__main__":
    main()
