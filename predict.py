from ultralytics import YOLO
import cv2

# Load a pretrained YOLOv8n model
model = YOLO('weights/best_fp16.engine', task='segment')

# Run inference on 'bus.jpg' with arguments
#results = model.predict('datasets/test.mp4', save=False, imgsz=640, conf=0.4, show=False, stream=True, show_boxes=False)
cap = cv2.VideoCapture('datasets/test.mp4')

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Inference", annotated_frame)