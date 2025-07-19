import cv2
from ultralytics import YOLO
import time

def detect_and_save_from_camera(model_path, camera_device='/dev/video0', output_path='output_camera.mp4', confidence_threshold=0.5):
    """
    Raspberry Pi veya Linux sistemde bir kameradan canlı görüntü alır,
    YOLO ile nesne tespiti yapar, anlık görüntüyü ekrana verir ve video kaydeder.

    Args:
        model_path (str): YOLO model dosyasının yolu.
        camera_device (str or int): Kamera aygıtı (/dev/video0 ya da 0 gibi).
        output_path (str): Kaydedilecek çıktı video dosyası.
        confidence_threshold (float): Tespitler için güven eşiği.
    """
    try:
        model = YOLO(model_path)
        print(f"YOLO modeli yüklendi: {model_path}")
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return

    cap = cv2.VideoCapture(camera_device)
    if not cap.isOpened():
        print(f"Kamera açılamadı: {camera_device}")
        return

    # Kamera özelliklerini al
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30  # bazı kameralar fps döndürmez

    print(f"Kamera açıldı: {camera_device} | {width}x{height} @ {fps:.2f} FPS")

    # VideoWriter ile çıktı dosyasını hazırlıyoruz
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kare alınamadı, çıkılıyor.")
            break

        # YOLO çıkarımı
        results = model(frame, conf=confidence_threshold, verbose=False)

        # Görsel olarak çizilmiş kare
        annotated_frame = results[0].plot()

        # Video dosyasına yaz
        out.write(annotated_frame)

        # Ekranda göster
        cv2.imshow("Kamera - YOLO Tespiti", annotated_frame)

        # 'q' tuşuna basıldığında çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            print(f"{frame_count} kare işlendi | Ortalama FPS: {frame_count / elapsed:.2f}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nİşlem tamamlandı. Toplam {frame_count} kare işlendi.")
    print(f"Çıktı dosyası: {output_path}")

# --- KULLANIM ---
if __name__ == "__main__":
    yolo_model_path = 'ihalow_ncnn_model'  # Eğer .pt modelin varsa örn: 'yolov8n.pt'
    camera_device_path = '/dev/video0'     # veya sadece 0 yazabilirsin
    output_video_path = 'camera_output.mp4'
    confidence = 0.5

    detect_and_save_from_camera(yolo_model_path, camera_device_path, output_video_path, confidence)
