"""Generate fake leaf images (one visual pattern per class) to test the pipeline end-to-end.
NOT for real results - use a real dataset (see download_kaggle.py)."""
import argparse, os, random
from PIL import Image, ImageDraw, ImageFilter
import config as C


def leaf_base(base, size=224):
    img = Image.new("RGB", (size, size), (235, 240, 235))
    d = ImageDraw.Draw(img)
    d.ellipse((20, 15, size - 20, size - 15), fill=base, outline=(20, 80, 20))
    d.line((30, size - 30, size - 30, 30), fill=(200, 220, 120), width=3)
    return img, d


def draw(cls, size=224):
    r = random.randint
    g = (r(40, 80), r(130, 170), r(40, 70))
    if cls == "nutrient_deficiency":
        g = (r(170, 210), r(200, 230), r(40, 80))
    img, d = leaf_base(g, size)
    c = size // 2
    if cls == "bacterial_leaf_spot":
        for _ in range(35): x, y = r(40, 180), r(40, 180); d.ellipse((x, y, x + r(3, 7), y + r(3, 7)), fill=(90, 50, 20))
    elif cls == "cercospora_leaf_spot":
        for _ in range(6):
            x, y = r(50, 170), r(50, 170); s = r(14, 24)
            d.ellipse((x, y, x + s, y + s), fill=(200, 170, 90), outline=(80, 50, 20), width=3)
    elif cls == "anthracnose":
        d.ellipse((c - 35, c - 25, c + 35, c + 30), fill=(90, 55, 25), outline=(220, 190, 40), width=6)
    elif cls == "powdery_mildew":
        for _ in range(30): x, y = r(30, 190), r(30, 190); d.ellipse((x, y, x + r(8, 20), y + r(8, 20)), fill=(240, 240, 240))
    elif cls == "downy_mildew":
        for _ in range(8): x, y = r(40, 170), r(40, 170); d.ellipse((x, y, x + r(20, 40), y + r(15, 30)), fill=(210, 220, 70))
    elif cls == "mosaic_virus":
        for _ in range(50): x, y = r(30, 190), r(30, 190); d.ellipse((x, y, x + 14, y + 14), fill=(r(60, 200), r(170, 230), 50))
    elif cls == "leaf_curl_virus":
        img = img.filter(ImageFilter.GaussianBlur(1)); d = ImageDraw.Draw(img)
        for i in range(0, size, 12): d.arc((10, i, size - 10, i + 40), 0, 180, fill=(20, 100, 20), width=2)
    elif cls == "leaf_blight":
        d.polygon([(30, 190), (30, 80), (110, 130), (150, 190)], fill=(110, 70, 30))
    # healthy / nutrient_deficiency: plain
    px = img.load()
    for _ in range(300):  # noise
        px[r(0, size - 1), r(0, size - 1)] = (r(0, 255), r(0, 255), r(0, 255))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=60)
    args = ap.parse_args()
    random.seed(C.SEED)
    for cls in C.CLASS_LABELS:
        out = os.path.join(C.ALL_DIR, cls)
        os.makedirs(out, exist_ok=True)
        for i in range(args.per_class):
            draw(cls).save(os.path.join(out, f"{cls}_{i:04d}.jpg"), quality=90)
        print("generated", cls)


if __name__ == "__main__":
    main()
