from ultralytics import YOLO

model = YOLO('yolov8m.pt')

model.train(data='custom_data.yaml', epochs=100)
