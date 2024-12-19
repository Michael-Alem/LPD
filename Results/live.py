from ultralytics import YOLO
import cv2

# Load the pretrained YOLO model
model = YOLO('D:\\license_plate_detection\\license_plate_detection.pt')

# Run inference on the source (webcam)
results = model(source=0, show=True, conf=0.3, save=True)
