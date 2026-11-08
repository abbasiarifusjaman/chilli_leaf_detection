# Chilli Leaf Disease Detection (Custom CNN)

Classifies chilli leaves into 10 classes: Healthy, Bacterial Leaf Spot, Cercospora Leaf Spot,
Anthracnose, Powdery Mildew, Downy Mildew, Mosaic Virus, Leaf Curl Virus, Leaf Blight, Nutrient Deficiency.

## Architecture
Input 224×224×3 → Conv(32,3×3)+ReLU → MaxPool(2×2) → Conv(64)+ReLU → MaxPool → Conv(128)+ReLU → MaxPool
→ Flatten (100352) → Dense(256, ReLU) → Dropout(0.5) → Dense(128, ReLU) → Dense(10, Softmax)

## Setup

Use 64-bit Python 3.10–3.12. TensorFlow does not support the Python 3.14 32-bit
installation or Python 3.15 environment shown by some launchers.

### Windows PowerShell

Install 64-bit Python 3.12 first if `py -3.12 --version` does not work. Then run
these commands from the project folder. Calling Python through the virtual
environment avoids relying on `python`, `pip`, or `streamlit` being on `PATH`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the pipeline and web UI with the same environment:

```powershell
.\.venv\Scripts\python.exe generate_synthetic.py --per-class 60
.\.venv\Scripts\python.exe split_dataset.py
.\.venv\Scripts\python.exe train.py --epochs 5
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Option A – quick pipeline test (synthetic data, not real accuracy)
```bash
python generate_synthetic.py --per-class 60
python split_dataset.py
python train.py --epochs 5
python evaluate.py
```

## Option B – real data
```bash
python download_kaggle.py --dataset <owner/dataset-slug>     # or copy images into dataset/all/<class>/
python split_dataset.py                                       # 70/15/15
python train.py
python evaluate.py
```
Class folders inside `dataset/all/`: healthy, bacterial_leaf_spot, cercospora_leaf_spot, anthracnose,
powdery_mildew, downy_mildew, mosaic_virus, leaf_curl_virus, leaf_blight, nutrient_deficiency.

## Predict / Web UI
```bash
python predict.py path/to/leaf.jpg
streamlit run app.py
```

## Files
| File | Purpose |
|---|---|
| config.py | paths, hyperparameters, class names, treatment text |
| train.py | builds and trains the CNN, saves `models/chilli_cnn.keras` |
| evaluate.py | accuracy, precision, recall, F1, confusion matrix |
| predict.py | single-image prediction + treatment |
| app.py | Streamlit UI |
| split_dataset.py | 70/15/15 split |
| generate_synthetic.py | fake data for pipeline testing |
| download_kaggle.py | Kaggle downloader + folder mapping |

Treatment suggestions are general guidance; consult a local agricultural officer before applying chemicals.
