from ultralytics import YOLO

# Kendi modelinizi belirtin: "ihalow.pt"
model_name = "ihalow.pt"
model = YOLO(model_name)

# Modelinizi 480x480 boyutu ve FP16 (half=True) niceleme ile export edin
print(f"'{model_name}' modeli, 480x480 (FP16) NCNN formatına export ediliyor...")

model.export(
    format="ncnn", 
    imgsz=480, 
    half=True
)

print("Export tamamlandı!")
print(f"Oluşturulan klasör: '{model_name.split('.')[0]}_ncnn_model'")
