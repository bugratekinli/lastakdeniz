import cv2
from ultralytics import YOLO
import numpy as np
import time

def is_red_area(roi, red_thresh=0.3):
    """
    ROI (Region of Interest) içindeki kırmızı piksellerin oranını hesaplar.
    Eğer kırmızı oranı red_thresh'den büyükse True döner.
    """
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Kırmızı renk için HSV aralıkları (alt ve üst)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    red_ratio = cv2.countNonZero(red_mask) / (roi.shape[0] * roi.shape[1])

    return red_ratio > red_thresh

def test_yolo_on_video_red_filter(model_path, video_path, confidence_threshold=0.5, output_path='output_red_filtered.avi'):
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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, original_fps, (width, height))
    print(f"Video kaydı başlatıldı: '{output_path}'")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = model(frame, conf=confidence_threshold, verbose=False)

        # Yeni kareyi tamamen siyah yapalım (detekte edilen kırmızı nesneler burada gösterilecek)
        filtered_frame = np.zeros_like(frame)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box koordinatları
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                if is_red_area(roi, red_thresh=0.3):
                    # Eğer kutu içindeki alanın %30'undan fazlası kırmızı ise, orijinal görüntüyü kopyala
                    filtered_frame[y1:y2, x1:x2] = roi

        out.write(filtered_frame)

        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            print(f"Kare: {frame_count}, Anlık Ham Çıkarım FPS: {current_fps:.2f}")

    cap.release()
    out.release()

    total_elapsed_time = time.time() - start_time
    average_fps = frame_count / total_elapsed_time if total_elapsed_time > 0 else 0

    print(f"\nVideo işleme tamamlandı ve '{output_path}' dosyasına kaydedildi.")
    print(f"Toplam {frame_count} kare işlendi.")
    print(f"Ortalama Ham Çıkarım FPS: {average_fps:.2f}")

# --- KULLANIM ---
if __name__ == "__main__":
    yolo_model_path = 'ihalow_ncnn_model'
    input_video_file = '/dev/video0'
    confidence = 0.5
    output_file = 'ihavideo_red_filtered_output.avi'

    test_yolo_on_video_red_filter(yolo_model_path, input_video_file, confidence, output_file)
