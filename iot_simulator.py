import time
import requests
import cv2

SERVER_URL = "http://127.0.0.1:8000/api/iot/upload"
DEVICE_ID = "CAMERA_GANTRY_01"

def send_image_from_file(image_path):
    """Kirim file gambar yang ada di disk"""
    with open(image_path, "rb") as f:
        files = {"file": (image_path, f, "image/jpeg")}
        params = {"device_id": DEVICE_ID}
        response = requests.post(SERVER_URL, files=files, params=params)
        print("Response:", response.json())

def capture_and_send_webcam():
    """Jepret webcam secara berkala lalu kirim ke server (mirip kerja IoT)"""
    cap = cv2.VideoCapture(0)
    print("Membuka kamera IoT... Tekan Ctrl+C untuk berhenti.")
    
    try:
        while True:
            ret, frame = cap.read()
            if ret:
                _, img_encoded = cv2.imencode(".jpg", frame)
                files = {"file": ("capture.jpg", img_encoded.tobytes(), "image/jpeg")}
                params = {"device_id": DEVICE_ID}
                try:
                    res = requests.post(SERVER_URL, files=files, params=params, timeout=5)
                    print(f"[{time.strftime('%X')}] Terkirim: {res.json()}")
                except Exception as e:
                    print("Gagal kirim ke server:", e)
            
            # Ambil foto setiap 5 detik
            time.sleep(5)
    except KeyboardInterrupt:
        cap.release()
        print("Kamera IoT dihentikan.")

if __name__ == "__main__":
    # Opsi 1: Jika ingin tes pakai webcam laptop/PC
    capture_and_send_webcam()

    # Opsi 2: Jika ingin tes pakai file foto pipa yang sudah ada
    # send_image_from_file("foto_tumpukan_pipa.jpg")