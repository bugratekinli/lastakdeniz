from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
model = YOLO("ihalow.pt")

# Export the model to NCNN format
model.export(format="ncnn", imgsz=640)


