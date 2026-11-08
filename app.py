"""Streamlit UI:  streamlit run app.py"""
import streamlit as st
from PIL import Image
import config as C
from predict import load_model_and_classes, predict_image

st.set_page_config(page_title="Chilli Leaf Disease Detector", page_icon="🌶️")
st.title("🌶️ Chilli Leaf Disease Detection")
st.caption("Upload a chilli leaf photo to identify the disease and get treatment advice.")


@st.cache_resource
def get_model():
    return load_model_and_classes()


try:
    model, classes = get_model()
except Exception as e:
    st.error(f"Model not found. Train first with `python train.py`.\n\n{e}")
    st.stop()

file = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded leaf", use_container_width=True)
    results = predict_image(model, classes, img, top_k=3)
    best, conf = results[0]
    st.subheader(f"{C.pretty(best)} — {conf*100:.1f}%")
    st.info(C.TREATMENTS.get(best, "Consult a local agricultural expert."))
    st.write("**Top predictions**")
    for c, p in results:
        st.progress(min(max(p, 0.0), 1.0), text=f"{C.pretty(c)}: {p*100:.1f}%")
