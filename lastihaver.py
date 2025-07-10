import cv2
from ultralytics import YOLO
import numpy as np
import time # Zaman ölçümü için

def test_yolo_on_video(model_path, video_path, output_path="output_video.avi", confidence_threshold=0.5):
    """
    YOLO modelini bir video üzerinde test eder ve tespit edilen nesneleri içeren yeni bir video kaydeder.
    Ek olarak, orijinal videonun FPS'ini ve işlem sırasında anlık FPS'i gösterir.

    Args:
        model_path (str): YOLO model ağırlık dosyasının yolu (örn: 'yolov8n.pt').
        video_path (str): Test edilecek video dosyasının yolu (örn: 'input_video.mp4').
        output_path (str): Sonuç videosunun kaydedileceği yol (örn: 'output_video.avi').
        confidence_threshold (float): Tespitler için güven eşiği. Bu eşiğin altındaki tespitler gösterilmez.
    """
    try:
        # YOLO modelini yükle
        model = YOLO(model_path)
        print(f"YOLO modeli '{model_path}' başarıyla yüklendi.")
    except Exception as e:
        print(f"Hata: YOLO modeli yüklenirken bir sorun oluştu: {e}")
        return

    # Video dosyasını aç
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Hata: Video dosyası '{video_path}' açılamadı.")
        return

    # Video özelliklerini al
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS) # Orijinal videonun FPS'i
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Çıktı video formatı (örn: MJPG, XVID)

    # Video yazıcı nesnesini oluştur
    out = cv2.VideoWriter(output_path, fourcc, int(original_fps), (frame_width, frame_height))

    print(f"Video işleniyor: '{video_path}'...")
    print(f"Orijinal video FPS: {original_fps:.2f}")

    frame_count = 0
    start_time = time.time() # FPS hesaplaması için başlangıç zamanı

    while True:
        ret, frame = cap.read() # Kare oku
        if not ret:
            break # Video bittiğinde döngüden çık

        frame_count += 1

        # Kare üzerinde nesne tespiti yap
        results = model(frame, conf=confidence_threshold, verbose=False)

        # Tespit sonuçlarını görselleştir
        annotated_frame = results[0].plot()

        # İşlenmiş kareyi çıktı videosuna yaz
        out.write(annotated_frame)

        # Anlık FPS hesaplaması
        end_time = time.time()
        elapsed_time = end_time - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Konsola FPS çıktısı ver
        # Her 30 karede bir veya belirli aralıklarla çıktı vermek performansı etkilemez
        if frame_count % 30 == 0: # Her 30 karede bir güncellenmiş FPS'i göster
            print(f"Kare: {frame_count}, Anlık İşleme FPS: {current_fps:.2f}")

        # Sonuç karesini göster (isteğe bağlı, Raspberry Pi'de yavaşlatabilir)
        # cv2.imshow("YOLO Object Detection", annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # Kaynakları serbest bırak
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Genel ortalama FPS hesaplaması
    total_elapsed_time = time.time() - start_time
    average_fps = frame_count / total_elapsed_time if total_elapsed_time > 0 else 0

    print(f"\nVideo işleme tamamlandı. Sonuç '{output_path}' konumuna kaydedildi.")
    print(f"Toplam {frame_count} kare işlendi.")
    print(f"Ortalama İşleme FPS: {average_fps:.2f}")


# --- KULLANIM ÖRNEĞİ ---
if __name__ == "__main__":
    # Kendi model dosyanızın yolu
    yolo_model_path = 'iha.pt'

    # Test etmek istediğiniz video dosyasının yolu
    input_video_file = 'ihavideo.mp4'

    # Çıktı videosunun kaydedileceği yol
    output_video_file = 'output_detections_with_fps.avi'

    # Güven eşiği (0.0 ile 1.0 arasında)
    confidence = 0.5

    # Fonksiyonu çağır
    test_yolo_on_video(yolo_model_path, input_video_file, output_video_file, confidence)
