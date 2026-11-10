import os
import cv2
import numpy as np
import base64
import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse

app = FastAPI(title="Industrial Pipe Tally - Dual Mode Complete")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(BASE_DIR, "templates", "index.html")

def detect_pipes_core(img, crop_box=None):
    """Deteksi kontur rongga elips & lingkaran di seluruh gambar atau area crop"""
    img_h, img_w = img.shape[:2]
    
    # Cek apakah ada kotak crop
    if crop_box and crop_box.get("w", 0) > 30 and crop_box.get("h", 0) > 30:
        bx = max(0, int(crop_box["x"]))
        by = max(0, int(crop_box["y"]))
        bw = min(img_w - bx, int(crop_box["w"]))
        bh = min(img_h - by, int(crop_box["h"]))
        crop = img[by:by+bh, bx:bx+bw]
    else:
        bx, by = 0, 0
        crop = img

    ch, cw = crop.shape[:2]
    total_area = ch * cw
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.medianBlur(enhanced, 5)

    block_size = int(min(ch, cw) * 0.08)
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(15, block_size)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 5
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    min_area = total_area * 0.001
    max_area = total_area * 0.12

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            (cx, cy), (ma, ma_major), angle = ellipse
            if ma == 0 or ma_major == 0:
                continue

            aspect_ratio = min(ma, ma_major) / max(ma, ma_major)
            if aspect_ratio >= 0.40:
                avg_r = int((ma + ma_major) / 4)
                candidates.append({
                    "x": int(cx) + bx,
                    "y": int(cy) + by,
                    "r": max(12, avg_r)
                })

    if not candidates:
        return [], 30

    median_r = np.median([c["r"] for c in candidates])
    min_dist = median_r * 1.35
    filtered = []
    candidates.sort(key=lambda c: abs(c["r"] - median_r))

    for c in candidates:
        if not any(np.hypot(c["x"] - f["x"], c["y"] - f["y"]) < min_dist for f in filtered):
            c["r"] = int(median_r)
            filtered.append(c)

    # Urutkan Atas ke Bawah, Kiri ke Kanan
    row_height = max(15, int(median_r * 1.5))
    filtered.sort(key=lambda c: (int(c["y"] // row_height), c["x"]))

    return filtered, int(median_r)

@app.get("/")
async def index():
    if not os.path.exists(HTML_PATH):
        return HTMLResponse(f"<h3>Template tidak ditemukan di: {HTML_PATH}</h3>")
    return FileResponse(HTML_PATH)

@app.post("/api/upload-raw")
async def upload_raw(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Gambar rusak"})

    _, raw_buf = cv2.imencode('.jpg', img)
    raw_b64 = base64.b64encode(raw_buf).decode('utf-8')

    return JSONResponse(content={
        "status": "success",
        "raw_image": f"data:image/jpeg;base64,{raw_b64}"
    })

@app.post("/api/process-cv")
async def process_cv(
    file: UploadFile = File(...),
    crop_box: str = Form(None)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Gambar rusak"})

    box_data = json.loads(crop_box) if crop_box else None
    pipes, dominant_r = detect_pipes_core(img, box_data)

    return JSONResponse(content={
        "status": "success",
        "total": len(pipes),
        "circles": pipes,
        "dominant_radius": dominant_r
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)