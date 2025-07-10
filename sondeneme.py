import cv2
import numpy as np
import ncnn
import time

def test_yolo_ncnn_fps_and_save_video(param_path, bin_path, video_file_path, output_video_path="output_detections.avi"):
    """
    NCNN ile YOLO modelini bir video dosyasında çalıştırır, FPS'i ölçer ve
    tespitleri içeren yeni bir video kaydeder.
    Görüntüleme ve kare yeniden boyutlandırma işlemleri yapılmaz.
    UYARI: Modelin, doğrudan video karelerinin boyutunu kabul ettiğinden emin olun.

    Args:
        param_path (str): NCNN model .param dosyasının yolu.
        bin_path (str): NCNN model .bin dosyasının yolu.
        video_file_path (str): Test edilecek video dosyasının yolu (örn: 'ihavideo.mp4').
        output_video_path (str): İşlenmiş videonun kaydedileceği yol (örn: 'output_detections.avi').
    """
    try:
        net = ncnn.Net()
        # NCNN thread sayısını ayarlama (CPU çekirdeklerini kullanmak için)
        net.opt.num_threads = 4 # Raspberry Pi 5'in çekirdek sayısına göre ayarlayın (4 veya 8)
        # net.opt.num_threads = cv2.getBuildInformation().count("TBB") # Alternatif
        
        net.load_param(param_path)
        net.load_model(bin_path)
        print(f"NCNN modeli '{param_path}' ve '{bin_path}' başarıyla yüklendi.")
    except Exception as e:
        print(f"Hata: NCNN modeli yüklenirken bir sorun oluştu: {e}")
        return

    cap = cv2.VideoCapture(video_file_path)

    if not cap.isOpened():
        print(f"Hata: Video dosyası '{video_file_path}' açılamadı.")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video işleniyor: '{video_file_path}' (FPS ölçülecek ve sonuç kaydedilecek)...")
    print(f"Orijinal video boyutu: {original_width}x{original_height}, FPS: {original_fps:.2f}")

    # Modelin beklediği giriş boyutunu doğrudan kontrol edemediğimiz için uyarı verelim.
    print(f"UYARI: Video kareleri doğrudan NCNN modeline beslenecektir ({original_width}x{original_height}).")
    print("Modelin bu boyutta giriş beklediğinden emin olun, aksi takdirde hata alabilirsiniz.")

    # Video yazıcı nesnesini oluştur
    # MJPG codec genellikle iyi bir uyumluluk sunar.
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, int(original_fps), (original_width, original_height))

    if not out.isOpened():
        print(f"Hata: Çıkış videosu '{output_video_path}' yazılamadı. Codec veya yol hatası olabilir.")
        cap.release()
        return

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break # Video bittiğinde veya hata oluştuğunda çık

        frame_count += 1

        # Görüntüyü BGR'den RGB'ye dönüştür ve normalleştir (NCNN giriş formatına uygun)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame_float = rgb_frame.astype(np.float32) / 255.0

        # NumPy dizisini NCNN Mat'e dönüştür
        mat = ncnn.Mat.from_pixels(rgb_frame_float, ncnn.Mat.PixelType.PIXEL_RGB, original_width, original_height)

        # Modelde çıkarım yap
        ex = net.create_extractor()
        ex.input("images", mat)
        ret, output = ex.extract("output0")

        # <<< BURADA TESİP EDİLEN KUTULARI VE ETİKETLERİ 'frame' ÜZERİNE ÇİZMENİZ GEREKİR >>>
        # NCNN çıktısı 'output' Mat'inden bounding box ve sınıf bilgilerini
        # ayrıştırmanız ve ardından 'cv2.rectangle', 'cv2.putText' gibi OpenCV fonksiyonlarıyla
        # 'frame' üzerine çizmeniz gerekiyor.
        # Ultralytics'in 'plot()' metodu burada NCNN çıktısıyla direkt çalışmaz.
        # Bu kısım modelinizin çıktısına göre değişecektir.
        # Şimdilik, çizim yapmadan direkt orijinal kareyi kaydedeceğiz veya boş bir kare oluşturup kaydedebiliriz.
        # Eğer çizim yapmak isterseniz, buraya özel kod eklemelisiniz.
        # Basitçe orijinal kareyi kaydedelim veya modelin çıktılarını işlemeye karar verelim:
        processed_frame = frame.copy() # Orijinal kareyi değiştirmeden bir kopyasını al
        # ... Eğer tespitleri çiziyorsanız, çizim kodunuz buraya gelecek ...
        # Örnek:
        # for detection in parse_ncnn_output(output, original_width, original_height):
        #     x1, y1, x2, y2, conf, cls = detection
        #     cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(processed_frame, f"{cls}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # <<< YUKARIDAKİ KISIM MODELİN ÇIKTISINI İŞLEMEYE YÖNELİK ÖRNEKTİR >>>


        # İşlenmiş kareyi çıktı videosuna yaz
        out.write(processed_frame)

        # Anlık FPS hesaplaması
        current_time = time.time()
        elapsed_time = current_time - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        if frame_count % 30 == 0:
            print(f"Kare: {frame_count}, Anlık Çıkarım+Kaydetme FPS: {current_fps:.2f}")

    cap.release()
    out.release() # Video yazıcıyı serbest bırak

    total_elapsed_time = time.time() - start_time
    average_fps = frame_count / total_elapsed_time if total_elapsed_time > 0 else 0

    print(f"\nİşleme tamamlandı (NCNN çıkarımı ve video kaydı).")
    print(f"Sonuç videosu '{output_video_path}' konumuna kaydedildi.")
    print(f"Toplam {frame_count} kare işlendi.")
    print(f"Ortalama Çıkarım+Kaydetme FPS: {average_fps:.2f}")

# --- KULLANIM ÖRNEĞİ ---
if __name__ == "__main__":
    param_file = "iha_ncnn_model/model.ncnn.param"
    bin_file = "iha_ncnn_model/model.ncnn.bin"

    video_to_test = 'ihavideo.mp4'
    output_video_to_save = 'output_detections.avi' # Kaydedilecek videonun adı

    test_yolo_ncnn_fps_and_save_video(param_file, bin_file, video_to_test, output_video_to_save)
