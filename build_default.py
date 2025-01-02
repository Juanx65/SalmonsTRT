from ultralytics import YOLO

model = YOLO('weights/yolov11salmons.pt')  # load a custom trained model

# Export the model
model.export(format='engine', int8=False, half=False,dynamic=True,batch=128,imgsz=640, data='datasets/salmons/salmons.yaml')