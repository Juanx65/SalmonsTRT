from ultralytics import YOLO
import cv2

# Load a pretrained YOLOv8n model
model = YOLO('weights/best.pt', task='segment')

# Run inference on 'bus.jpg' with arguments
#model.predict('datasets/test.mp4', save=False, conf=0.4, show=True, stream=False)
model.predict('datasets/salmons/images/val/Img3.jpeg', save=True, conf=0.4, show=True, stream=False,show_boxes=False)


""" 
cap = cv2.VideoCapture('datasets/test.mp4')

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if success:
        model(frame, show=True,conf=0.5)

 """