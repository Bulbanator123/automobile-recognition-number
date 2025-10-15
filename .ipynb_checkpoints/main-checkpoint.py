from ultralytics import YOLO

model = YOLO(model="yolov8m.pt")

results = model.train(data="datasets/data.yaml", epochs=100, imgsz=640, model="yolov8m.pt", multi_scale=True)