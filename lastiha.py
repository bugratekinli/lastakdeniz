import cv2
from ultralytics import YOLO
import numpy as np

def test_yolo_on_video(model_path, video_path, output_path="ihavideo.mp4", confidence_threshold=0.5):
    """
    YOLO modelini bir video üzerinde test eder ve tespit edilen nesneleri içeren yeni bir video kaydeder.

    Args:
        model_path (str): YOLO model ağırlık dosyasının yolu (örn: 'yolov8n.pt').
        video_path (str): Test edilecek video dosyasının yolu (örn: 'input_video.mp4').
        output_path (str): Sonuç videosunun kaydedileceği yol (örn: 'output_video.avi').
        confidence_threshold (float): Tespitler için güven eşiği. Bu eşiğin altındaki tespitler gösterilmez.
    """
    try:
        # YOLO modelini yükle
        # Eğer model_path 'yolov8n.pt' gibi bir isimse, kütüphane otomatik indirir.
        # Kendi eğitilmiş modeliniz ise tam yolunu belirtin.
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
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Çıktı video formatı (örn: MJPG, XVID)

    # Video yazıcı nesnesini oluştur
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Video işleniyor: '{video_path}'...")
    frame_count = 0

    while True:
        ret, frame = cap.read() # Kare oku
        if not ret:
            break # Video bittiğinde döngüden çık

        frame_count += 1
        # print(f"Kare {frame_count} işleniyor...")

        # Kare üzerinde nesne tespiti yap
        # stream=True, büyük videolarda daha iyi performans sağlayabilir
        results = model(frame, conf=confidence_threshold, verbose=False) # verbose=False çıktıyı azaltır

        # Tespit sonuçlarını görselleştir
        # results[0].plot() metodu otomatik olarak kutuları ve etiketleri çizer
        annotated_frame = results[0].plot()

        # Sonuç karesini göster (isteğe bağlı)
        # cv2.imshow("YOLO Object Detection", annotated_frame)

        # İşlenmiş kareyi çıktı videosuna yaz
        out.write(annotated_frame)

        # 'q' tuşuna basıldığında çıkış yap
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Kaynakları serbest bırak
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video işleme tamamlandı. Sonuç '{output_path}' konumuna kaydedildi.")
    print(f"Toplam {frame_count} kare işlendi.")

# --- KULLANIM ÖRNEĞİ ---
if __name__ == "__main__":
    # Kendi model dosyanızın yolu (örn: 'yolov8n.pt' veya kendi eğitilmiş modeliniz: 'runs/detect/train/weights/best.pt')
    yolo_model_path = 'ihalow_ncnn_model'

    # Test etmek istediğiniz video dosyasının yolu
    input_video_file = 'ihavideo.mp4' # Örn: Masaüstünüzde 'input.mp4' adında bir video dosyası olmalı

    # Çıktı videosunun kaydedileceği yol
    output_video_file = 'output_detections.avi'

    # Güven eşiği (0.0 ile 1.0 arasında)
    confidence = 0.5

    # Fonksiyonu çağır
    test_yolo_on_video(yolo_model_path, input_video_file, output_video_file, confidence)
