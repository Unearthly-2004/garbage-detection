import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO


model = YOLO("yolo11n.pt")  # load an official model
model = YOLO("yolov11.pt")  # load a custom model


def make_prediction(img):
    prediction = model.predict(img)
    return prediction


## Dashboard
st.title("Garbage Detector :")
upload = st.file_uploader(label="Upload Image :", type=["png", "jpg", "jpeg"])  ## Image as Bytes

if upload:
    img = Image.open(upload)
    prediction = make_prediction(img)
    for result in prediction:

        if result.obb:
            st.write("Oriented Bounding Boxes detected:")
            st.write(result.obb)

        # Optionally, save and display the final image with results
        result.save(f"result_{upload.name}")
        result_image = Image.open(f"result_{upload.name}")
        st.image(result_image, caption=f"Result Image - {upload.name}")
