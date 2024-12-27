from ultralytics import YOLO

model = YOLO('weights/yolov8lsalmons.pt')  # load a custom trained model

# Export the model
model.export(format='engine', int8=True, half=False,dynamic=True,batch=32)