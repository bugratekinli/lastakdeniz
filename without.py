import cv2
from ultralytics import YOLO
import numpy as np
import time

def test_yolo_on_video_raw_inference(model_path, video_path, confidence_threshold=0.5):
    """
    YOLO modelini bir video üzerinde test ederken, sadece çıkarım (inference) hızını ölçer.
    Video kaydetme, ekranda gösterme ve çıktıları görselleştirme işlemleri devre dışıdır.

    Args:
        model_path (str): YOLO model ağırlık dosyasının yolu (örn: 'yolov8n.pt').
        video_path (str): Test edilecek video dosyasının yolu (örn: 'input_video.mp4').
        confidence_threshold (float): Tespitler için güven eşiği.
    """
    try:
        model = YOLO(model_path)
        print(f"YOLO modeli '{model_path}' başarıyla yüklendi.")
    except Exception as e:
        print(f"Hata: YOLO modeli yüklenirken bir sorun oluştu: {e}")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Hata: Video dosyası '{video_path}' açılamadı.")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video işleniyor: '{video_path}' (Sadece ham çıkarım hızı ölçülecek)...")
    print(f"Orijinal video FPS: {original_fps:.2f}")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Kare üzerinde nesne tespiti yap (saf çıkarım)
        # Sadece modelin kendisini çalıştırıyoruz, çıktıları işlemeye veya çizmeye gerek yok.
        results = model(frame, conf=confidence_threshold, verbose=False)

        # Anlık FPS hesaplaması
        end_time = time.time()
        elapsed_time = end_time - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        if frame_count % 30 == 0:
            print(f"Kare: {frame_count}, Anlık Ham Çıkarım FPS: {current_fps:.2f}")

    cap.release()

    total_elapsed_time = time.time() - start_time
    average_fps = frame_count / total_elapsed_time if total_elapsed_time > 0 else 0

    print(f"\nVideo işleme tamamlandı (ham çıkarım modu).")
    print(f"Toplam {frame_count} kare işlendi.")
    print(f"Ortalama Ham Çıkarım FPS: {average_fps:.2f}")

# --- KULLANIM ÖRNEĞİ ---
if __name__ == "__main__":
    yolo_model_path = 'ihalow_ncnn_model'
    input_video_file = 'ihavideo.mp4'
    confidence = 0.5

    test_yolo_on_video_raw_inference(yolo_model_path, input_video_file, confidence)
