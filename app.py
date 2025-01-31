import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import cv2
import os
import shutil

st.set_page_config(
   page_title="YOLOv8 Car License Plate Image Processing",
   page_icon=":car:",
   initial_sidebar_state="expanded",
)
st.title('YOLO Car License Plate :green[Image Processing]')

# Allow users to upload images
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Load YOLO model
try:
    model = YOLO('license_plate_detection.pt') 
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")

def predict_and_save_image(path_test_car:str, output_image_path:str)-> str:
    """
    Predicts and saves the bounding boxes on the given test image using the trained YOLO model.
    
    Parameters:
    path_test_car (str): Path to the test image file.
    output_image_path (str): Path to save the output image file.

    Returns:
    str: The path to the saved output image file.
    """
    try:
        results = model.predict(path_test_car)
        image = cv2.imread(path_test_car)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f'license-plate {confidence:.2f}%'
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                cv2.rectangle(image, (x1, y1 - h - 5), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(image, text, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        # Save the image
        cv2.imwrite(output_image_path, image)
        return output_image_path
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def process_image(input_path:str, output_path:str) -> str:
    """
    Processes the uploaded image file and returns the path to the saved output file.

    Parameters:
    input_path (str): Path to the input image file.
    output_path (str): Path to save the output image file.

    Returns:
    str: The path to the saved output image file.
    """
    return predict_and_save_image(input_path, output_path)

temp_directory = 'temp'
if not os.path.exists(temp_directory):
    os.makedirs(temp_directory)

if st.button("Proceed"):
    if uploaded_file is not None:
        input_path = os.path.join("temp", uploaded_file.name)
        output_path = os.path.join("temp", f"output_{uploaded_file.name}")
        try:
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner('Processing...'):
                result_path = process_image(input_path, output_path)
                if result_path:
                    st.image(result_path)
        except Exception as e:
            st.error(f"Error uploading or processing file: {e}")
