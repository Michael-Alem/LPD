import streamlit as st
import os
import pytesseract
from ultralytics import YOLO
import torch

def inference(
    path2img: str,
    show_img: bool = False,
    size_img: int = 1080,
    nms_conf_thresh: float = 0.7,
    max_detect: int = 10,
) -> torch.Tensor:
    model = YOLO.load("Mike-Alem/License-Plate-Detection-First-Project")

    model.conf = nms_conf_thresh
    model.iou = 0.45
    model.agnostic = False
    model.multi_label = False
    model.max_det = max_detect

    results = model(path2img, size=size_img)
    results = model(path2img, augment=True)

    if show_img:
        results.show()

    return results.pred[0]

def ocr_tesseract(path2img):
 text = pytesseract.image_to_string(
 path2img,
 lang="eng",
 config="--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
 )
 return text

def app():
    st.header("License Plate Recognition Web App")
    st.subheader("Powered by YOLOv8")
    st.write("Welcome!")

    # add file uploader
    with st.form("my_uploader"):
        uploaded_file = st.file_uploader(
            "Upload image", type=["png", "jpg", "jpeg"], accept_multiple_files=False
        )
        submit = st.form_submit_button(label="Upload")

    # Corrected indentation for the following block
    if uploaded_file is not None:
        # save uploaded image
        save_path = os.path.join("temp", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if submit:
            text = run_license_plate_recognition(save_path).recognize_text()
            st.write(f"Detected License Plate Number: {text}")


if __name__ == "__main__":
    app()