"""Split dataset/all/<class>/* into train/val/test (default 70/15/15)."""
import argparse, os, random, shutil
import config as C

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    args = ap.parse_args()
    random.seed(C.SEED)

    for split in (C.TRAIN_DIR, C.VAL_DIR, C.TEST_DIR):
        for d in os.listdir(split) if os.path.isdir(split) else []:
            p = os.path.join(split, d)
            if os.path.isdir(p):
                shutil.rmtree(p)

    classes = [d for d in sorted(os.listdir(C.ALL_DIR)) if os.path.isdir(os.path.join(C.ALL_DIR, d))]
    if not classes:
        raise SystemExit("dataset/all is empty. Add images in dataset/all/<class_name>/")

    for cls in classes:
        files = [f for f in os.listdir(os.path.join(C.ALL_DIR, cls)) if f.lower().endswith(EXT)]
        random.shuffle(files)
        n = len(files)
        n_tr, n_va = int(n * args.train), int(n * args.val)
        parts = {C.TRAIN_DIR: files[:n_tr], C.VAL_DIR: files[n_tr:n_tr + n_va], C.TEST_DIR: files[n_tr + n_va:]}
        for dest, fl in parts.items():
            os.makedirs(os.path.join(dest, cls), exist_ok=True)
            for f in fl:
                shutil.copy2(os.path.join(C.ALL_DIR, cls, f), os.path.join(dest, cls, f))
        print(f"{cls:22s} total={n:4d} train={len(parts[C.TRAIN_DIR])} val={len(parts[C.VAL_DIR])} test={len(parts[C.TEST_DIR])}")


if __name__ == "__main__":
    main()
