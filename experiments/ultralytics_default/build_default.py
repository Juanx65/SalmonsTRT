from ultralytics import YOLO

model = YOLO('weights/yolov8lsalmons.pt')  # load a custom trained model

# Export the model
model.export(format='engine', int8=False, half=True,dynamic=True,batch=1,imgsz=640, data='datasets/salmons/salmons.yaml')