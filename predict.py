from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('weights/best_int8.engine', task='segment')

# Run inference on 'bus.jpg' with arguments
model.predict('datasets/test.mp4', save=True, imgsz=640, conf=0.4, show=True)