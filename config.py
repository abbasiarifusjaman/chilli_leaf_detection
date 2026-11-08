"""Shared settings, class names and treatment advice."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
ALL_DIR = os.path.join(DATASET_DIR, "all")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "chilli_cnn.keras")
CLASSES_PATH = os.path.join(MODEL_DIR, "class_names.json")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
SEED = 42

# Folder name -> display name
CLASS_LABELS = {
    "healthy": "Healthy Leaf",
    "bacterial_leaf_spot": "Bacterial Leaf Spot",
    "cercospora_leaf_spot": "Cercospora Leaf Spot",
    "anthracnose": "Anthracnose",
    "powdery_mildew": "Powdery Mildew",
    "downy_mildew": "Downy Mildew",
    "mosaic_virus": "Chilli Mosaic Virus",
    "leaf_curl_virus": "Chilli Leaf Curl Virus",
    "leaf_blight": "Leaf Blight",
    "nutrient_deficiency": "Nutrient Deficiency",
}

TREATMENTS = {
    "healthy": "No disease detected. Continue regular watering, balanced fertilisation and monitoring.",
    "bacterial_leaf_spot": "Remove infected leaves; spray copper-based bactericide; avoid overhead irrigation; use disease-free seed and crop rotation.",
    "cercospora_leaf_spot": "Apply mancozeb or chlorothalonil fungicide; improve spacing and air flow; remove crop debris after harvest.",
    "anthracnose": "Spray carbendazim or azoxystrobin; remove infected fruits/leaves; use treated seed; avoid water splash.",
    "powdery_mildew": "Apply wettable sulphur or a systemic fungicide (e.g. hexaconazole); reduce humidity and ensure good ventilation.",
    "downy_mildew": "Use metalaxyl + mancozeb; avoid prolonged leaf wetness; improve drainage and spacing.",
    "mosaic_virus": "No cure. Rogue out infected plants; control aphid vectors with neem oil or imidacloprid; use virus-free seedlings.",
    "leaf_curl_virus": "No cure. Control whitefly vectors (yellow sticky traps, neem oil, thiamethoxam); remove infected plants; use resistant varieties.",
    "leaf_blight": "Apply a suitable fungicide (e.g. carbendazim, mancozeb); ensure drainage; rotate crops; remove infected tissue.",
    "nutrient_deficiency": "Get a soil test; apply balanced NPK and micronutrients (e.g. nitrogen, magnesium); maintain soil pH 6.0-6.8.",
}

def pretty(name: str) -> str:
    return CLASS_LABELS.get(name, name.replace("_", " ").title())
