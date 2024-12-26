from ultralytics import YOLO

model = YOLO('weights/yolov11salmons.pt')  # load a custom trained model

# Export the model
model.export(format='engine', int8=True, hald=False)