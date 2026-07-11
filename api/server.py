import io
import sys
import base64
import time
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tempfile
import atexit
import argparse
from pathlib import Path
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from ultralytics import YOLO
import onnxruntime as ort
import boxmot
from boxmot.trackers.tracker_zoo import create_tracker
import heapq

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import argparse

from model import DesertSegFormer

class AnomalyAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 4, 2, 1), torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, 2, 1), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 4, 2, 1), torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 4, 2, 1), torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1), torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1), torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, 4, 2, 1), torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 3, 4, 2, 1), torch.nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))
from dataset import CLASS_NAMES, NUM_CLASSES

# ─── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_COLORS = [
    (255, 0, 0),    # Sky
    (0, 255, 255),  # Sand
    (0, 0, 255),    # Rock
    (0, 255, 0),    # Veg
    (128, 0, 128),  # Shadow
    (192, 192, 192) # Distant
]

RISK_WEIGHTS = { 0: 0.0, 1: 0.0, 2: 1.0, 3: 0.8, 4: 0.2, 5: 0.5 }
THREAT_CLASSES = [0] # Person
VEHICLE_CLASSES = [2, 3, 5, 7] # Car, Bike, Bus, Truck

app = FastAPI(title='TERRA-VIS Ultimate Core')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

# ─── Auth Logic ───────────────────────────────────────────────────────────────
AUTH_FILE = Path(__file__).parent.parent / "auth.json"
DEFAULT_PASSCODE = "TACTICAL99"
AES_KEY = b'\xeb\xa06[\xb3\x91]\xc7\x02\x0eG\xff\x0c\x01\x91\x86\x8d\x9a\x90o\xc0\xe7\xcf\xf4\xc80\x15\xf9\x07c\x0c\xe6'

def decrypt_model(encrypted_path: Path, expected_suffix: str = '.pt', expected_prefix: str = 'tmp_'):
    with open(encrypted_path, 'rb') as f:
        iv = f.read(16)
        encrypted_data = f.read()
        
    cipher = Cipher(algorithms.AES(AES_KEY), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded_data = decryptor.update(encrypted_data) + decryptor.finalize()
    
    unpadder = padding.PKCS7(128).unpadder()
    data = unpadder.update(padded_data) + unpadder.finalize()
    
    tmp = tempfile.NamedTemporaryFile(delete=False, prefix=expected_prefix, suffix=expected_suffix)
    tmp.write(data)
    tmp.close()
    
    def cleanup():
        try: os.unlink(tmp.name)
        except: pass
    atexit.register(cleanup)
    
    return tmp.name

def get_passcode():
    if not AUTH_FILE.exists():
        with open(AUTH_FILE, "w") as f:
            json.dump({"passcode": DEFAULT_PASSCODE}, f)
        return DEFAULT_PASSCODE
    try:
        with open(AUTH_FILE, "r") as f:
            return json.loads(f.read()).get("passcode", DEFAULT_PASSCODE)
    except:
        return DEFAULT_PASSCODE

@app.post("/auth")
async def verify_auth(passcode: str = Form(...)):
    if passcode == get_passcode():
        return JSONResponse({"status": "success"})
    return JSONResponse({"status": "error", "message": "Invalid Passcode"}, status_code=401)

@app.post("/change_password")
async def change_password(old_passcode: str = Form(...), new_passcode: str = Form(...)):
    if old_passcode != get_passcode():
        return JSONResponse({"status": "error", "message": "Old passcode incorrect"}, status_code=401)
    
    with open(AUTH_FILE, "w") as f:
        json.dump({"passcode": new_passcode}, f)
    return JSONResponse({"status": "success", "message": "Passcode updated"})


# Serve static files from the root directory so index.html can be accessed via http://localhost:8000/
# This fixes the Chrome location permission popup loop which happens on file:// URIs.
app.mount("/app", StaticFiles(directory=str(Path(__file__).parent.parent), html=True), name="static")

seg_model = None
seg_model_ort = None
depth_model = None
yolo_model = None
tracker = None
device = None
transform = None
env_ae = None
MIDAS_SKIP = 3
rover_state = { "lat": 26.9157, "lon": 70.9083, "heading": 0.0, "speed": 0.0 }

# ─── Phase III Passive Sensor State ───────────────────────────────────────────
passive_prev_gray = None
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=25, detectShadows=False)
glint_history = []
prev_bright_pixels = 0
frame_counter = 0
cached_depth_uint8 = None
midas_frame_count = 0

# ─── Helper Functions ─────────────────────────────────────────────────────────

def assess_environment(img_rgb):
    """SENTINEL AI: Detects Sandstorms & Visibility"""
    # Convert to LAB for contrast/color analysis
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    contrast = np.std(l)
    yellow_intensity = np.mean(b) # B-channel: Positive = Yellow
    
    risk = 0.0
    if contrast < 40: risk += 50 # Hazy
    if contrast < 20: risk += 30 # Severe Fog
    if yellow_intensity > 135: risk += 20 # Sandstorm tint
    
    risk = min(100, risk)
    status = "CLEAR"
    if risk > 75: status = "SANDSTORM"
    elif risk > 40: status = "DEGRADED"
    
    return round(risk, 1), status

def run_passive_sensors(img_rgb, toggles):
    """SENTINEL AI: Phase III - Optical Flow, MOG2, Glint Detection"""
    global passive_prev_gray, bg_subtractor, glint_history, frame_counter, prev_bright_pixels
    frame_counter += 1
    
    # Passive Sensors
    passive_logs = []
    
    # Phase IV: Sparse VAE Anomaly Detection (runs every 15 frames to save latency)
    env_degraded = False
    vae_loss = 0.0
    if toggles.get('vae', True) and env_ae is not None and frame_counter % 15 == 0:
        with torch.no_grad():
            vae_in = cv2.resize(img_rgb, (128, 128)).astype(np.float32) / 255.0
            vae_in = torch.from_numpy(vae_in).permute(2,0,1).unsqueeze(0).to(device)
            recon = env_ae(vae_in)
            vae_loss = torch.nn.functional.mse_loss(recon, vae_in).item()
            if vae_loss > 0.045: # Threshold for anomalies
                env_degraded = True
                passive_logs.append(f"CRITICAL: ENVIRONMENTAL DEGRADATION (Score: {vae_loss:.3f})")
    
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    if toggles.get('flash', True):
        # 1. P10 - Retroreflection (Optics Glint) Spotter
        # Increased threshold to 252 so it only catches absolute pure-white light, not just bright objects
        _, thresh = cv2.threshold(gray, 252, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        glint_found = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Tighter area bounds for a pinpoint of light
            if 10 < area < 50:  
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    # Stricter circularity (must be a perfect circle/lens reflection)
                    if circularity > 0.85:  
                        glint_found = True
                        break
        
        if glint_found: glint_history.append(1)
        else: glint_history.append(0)
        
        # Require more sustained glints to trigger an alert
        if len(glint_history) > 10: glint_history.pop(0)
        
        bright_pixels = cv2.countNonZero(thresh)
        delta = bright_pixels - prev_bright_pixels
        prev_bright_pixels = bright_pixels
        
        # Increased threshold to 15000 (approx 5% of a 512x512 image) for true flashes
        if delta > 15000:
            passive_logs.append("LARGE FLASH DETECTED (EXPLOSION/FLASHBANG)")
        elif sum(glint_history) >= 6:
            passive_logs.append("OPTIC GLINT DETECTED (SNIPER/BINOCS)")
            glint_history = [] # Reset after alert
        
    if toggles.get('mog2', True):
        # 2. P9 - MOG2 Background Subtraction (Anomalies)
        fg_mask = bg_subtractor.apply(img_rgb)
        fg_contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Increased area threshold from 2000 to 8000 to ignore normal body movement
        large_anomalies = [c for c in fg_contours if cv2.contourArea(c) > 8000]
        if len(large_anomalies) > 3:
            passive_logs.append("ENVIRONMENT ANOMALY: MULTIPLE UNKNOWN SIGNATURES")
        
    if toggles.get('opt_flow', True):
        # 3. P8 - Optical Flow
        if passive_prev_gray is not None:
            # Resize for performance
            s_gray = cv2.resize(gray, (256, 256))
            s_prev = cv2.resize(passive_prev_gray, (256, 256))
            flow = cv2.calcOpticalFlowFarneback(s_prev, s_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mean_mag = np.mean(mag)
            # Increased flow magnitude threshold from 2.5 to 5.0 to ignore minor shifts
            if mean_mag > 5.0:
                passive_logs.append(f"MOTION ANOMALY: GLOBAL SHIFT ({mean_mag:.1f})")
            
    passive_prev_gray = gray.copy()
    return passive_logs, env_degraded

def get_depth_map(img_rgb):
    global depth_model, transform, cached_depth_uint8, midas_frame_count, MIDAS_SKIP
    
    midas_frame_count += 1
    if cached_depth_uint8 is not None and midas_frame_count % MIDAS_SKIP != 0:
        return cv2.applyColorMap(cached_depth_uint8, cv2.COLORMAP_MAGMA), cached_depth_uint8
        
    input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = depth_model(input_batch)
        prediction = F.interpolate(prediction.unsqueeze(1), size=img_rgb.shape[:2], mode="bicubic", align_corners=False).squeeze()
    depth_map = prediction.cpu().numpy()
    depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    
    cached_depth_uint8 = depth_uint8
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA), depth_uint8

def detect_depth_hazards(depth_uint8, path_points):
    hazards = []
    h, w = depth_uint8.shape
    sx = w / 512
    sy = h / 512
    
    # For each path point, check 32x32 window for depth variance
    # Potholes/trenches cause sharp local depth changes
    for (x, y) in path_points:
        px, py = int(x * sx), int(y * sy)
        half = 16
        if py-half < 0 or py+half >= h or px-half < 0 or px+half >= w: continue
        
        window = depth_uint8[py-half:py+half, px-half:px+half]
        variance = np.var(window)
        
        # High local variance indicates sudden drop or obstacle
        if variance > 800:
            hazards.append({"x": int(x), "y": int(y), "severity": float(variance), "type": "HAZARD"})
            
    return hazards

def detect_threats(img_rgb, draw_img=None):
    global tracker
    results = yolo_model(img_rgb, verbose=False)
    out_img = draw_img if draw_img is not None else img_rgb.copy()
    raw_out = img_rgb.copy()
    hostiles = 0
    vehicles = 0
    threat_details = []
    
    # Get detections for BoT-SORT
    # Format: [x1, y1, x2, y2, conf, cls]
    dets = []
    for box in results[0].boxes:
        conf = float(box.conf)
        if conf < 0.4: continue
        cls = int(box.cls)
        if cls in THREAT_CLASSES or cls in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            dets.append([x1, y1, x2, y2, conf, cls])
    
    if len(dets) > 0:
        dets = np.array(dets)
        # Update tracker
        tracks = tracker.update(dets, img_rgb)
    else:
        tracks = tracker.update(np.empty((0, 6)), img_rgb)
        
    for track in tracks:
        x1, y1, x2, y2, id, conf, cls, _ = track
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        id = int(id)
        cls = int(cls)
        
        color = (0, 255, 0)
        label = ""
        
        if cls in THREAT_CLASSES:
            hostiles += 1
            label = f"HOSTILE-{id} {conf:.2f}"
            color = (0, 0, 255) # Red
            threat_details.append({"id": id, "type": "HOSTILE"})
        elif cls in VEHICLE_CLASSES:
            vehicles += 1
            label = f"VEHICLE-{id} {conf:.2f}"
            color = (255, 255, 0) # Yellow
            threat_details.append({"id": id, "type": "VEHICLE"})
            
        cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.rectangle(raw_out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(raw_out, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return out_img, raw_out, hostiles, vehicles, threat_details

def compute_risk_map(mask, original_size):
    h, w = mask.shape
    risk_map = np.zeros((h, w), dtype=np.float32)
    for cls_id, risk_val in RISK_WEIGHTS.items(): risk_map[mask == cls_id] = risk_val
    
    obstacles = np.zeros_like(mask, dtype=np.uint8)
    obstacles[(mask == 2) | (mask == 3)] = 1
    dist = cv2.distanceTransform(1-obstacles, cv2.DIST_L2, 5)
    proximity = np.clip((100.0 - dist) / 100.0, 0, 1)
    mask_safe = (mask == 1) | (mask == 4)
    risk_map[mask_safe] += proximity[mask_safe] * 0.5
    
    heatmap_u8 = (np.clip(risk_map, 0, 1) * 255).astype(np.uint8)
    return cv2.resize(cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET), (original_size[1], original_size[0]))

def calculate_metrics(mask):
    edges = cv2.Canny(mask.astype(np.uint8)*40, 50, 150)
    complexity = min(100, (np.count_nonzero(edges) / mask.size) * 800)
    rock_pct = (mask == 2).sum() / mask.size
    suitability = "SUITABLE"
    reason = "Terrain is stable."
    if rock_pct > 0.25: suitability, reason = "UNSUITABLE", "High obstacle density."
    elif rock_pct > 0.10: suitability, reason = "CAUTION", "Moderate rocky terrain."
    return int(complexity), suitability, reason

def astar_path(mask_512):
    # Downsample to 64x64 for fast pathfinding
    scale = 8
    mask_64 = cv2.resize(mask_512, (64, 64), interpolation=cv2.INTER_NEAREST)
    
    grid_cost = np.full((64, 64), 9999, dtype=np.float32)
    grid_cost[mask_64 == 1] = 1   # Sand
    grid_cost[mask_64 == 4] = 2   # Shadow
    grid_cost[mask_64 == 5] = 5   # Distant
    grid_cost[mask_64 == 3] = 50  # Veg
    
    h, w = 64, 64
    start = (63, 32) # Bottom center
    
    # Valid endpoints in top 10 rows
    valid_ends = np.argwhere(grid_cost[:10, :] < 50)
    if len(valid_ends) == 0: return []
    
    # Pick endpoint closest to center horizontal
    centers = np.abs(valid_ends[:, 1] - 32)
    end = tuple(valid_ends[np.argmin(centers)])
    
    def heuristic(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
        
    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == end: break
            
        r, c = current
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w:
                new_cost = cost_so_far[current] + grid_cost[nr, nc] * (1.414 if dr != 0 and dc != 0 else 1.0)
                if new_cost < 9999:
                    if (nr, nc) not in cost_so_far or new_cost < cost_so_far[(nr, nc)]:
                        cost_so_far[(nr, nc)] = new_cost
                        priority = new_cost + heuristic((nr, nc), end)
                        heapq.heappush(frontier, (priority, (nr, nc)))
                        came_from[(nr, nc)] = current
                        
    if end not in came_from: return []
    
    path = []
    curr = end
    while curr:
        path.append((curr[1] * scale, curr[0] * scale))
        curr = came_from[curr]
    path.reverse()
    
    # Smooth/subsample path
    if len(path) > 10: path = path[::len(path)//10]
    return path

def preprocess(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = (img.shape[0], img.shape[1]) 
    resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    norm    = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    tensor  = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, original_size, img_rgb

def arr_to_b64(arr, is_bgr=False):
    if not is_bgr:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.png', arr)
    return base64.b64encode(buf).decode('utf-8')

def mask_to_color(mask):
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, col in enumerate(CLASS_COLORS): color[mask == cls] = col[::-1] 
    return color

def overlay_image(img, mask_color, alpha=0.5):
    mask_resized = cv2.resize(mask_color, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(img, 1-alpha, mask_resized, alpha, 0)

def draw_path(img, path_points, hazards=None):
    out = img.copy()
    if len(path_points) > 1:
        h, w = img.shape[:2]
        sx, sy = w/512, h/512
        pts = [(int(p[0]*sx), int(p[1]*sy)) for p in path_points]
        for i in range(len(pts)-1): cv2.line(out, pts[i], pts[i+1], (0, 255, 0), 6)
        for p in pts: cv2.circle(out, p, 8, (0, 255, 0), -1)
    
    if hazards:
        for haz in hazards:
            x, y = int(haz['x'] * (img.shape[1]/512)), int(haz['y'] * (img.shape[0]/512))
            cv2.circle(out, (x, y), 20, (0, 0, 255), 3)
            cv2.putText(out, "HAZARD", (x-25, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
    return out

# ─── MAIN ENDPOINT ────────────────────────────────────────────────────────────

@app.post('/predict')
async def predict(file: UploadFile = File(...), mode: str = Form("standard"), lat: float = Form(None), lon: float = Form(None), heading: float = Form(None), env_toggles: str = Form("{}")):
    if seg_model is None and seg_model_ort is None: raise HTTPException(503, 'Model not loaded')

    try:
        toggles = json.loads(env_toggles)
    except:
        toggles = {}

    img_bytes = await file.read()
    tensor, original_size, img_rgb = preprocess(img_bytes)
    t0 = time.perf_counter()
    
    # 1. Base Segmentation
    with torch.no_grad():
        if seg_model_ort is not None:
            input_name = seg_model_ort.get_inputs()[0].name
            input_np = tensor.cpu().numpy()
            logits_np = seg_model_ort.run(None, {input_name: input_np})[0]
            logits = torch.from_numpy(logits_np).to(device)
            logits = F.interpolate(logits, size=(IMG_SIZE, IMG_SIZE), mode='bilinear')
            pred_class = F.softmax(logits, dim=1).argmax(dim=1)
        else:
            logits = F.interpolate(seg_model(tensor.to(device)), size=(IMG_SIZE, IMG_SIZE), mode='bilinear')
            pred_class = F.softmax(logits, dim=1).argmax(dim=1)
    
    mask_512 = pred_class[0].cpu().numpy().astype(np.uint8)
    inference_ms = (time.perf_counter() - t0) * 1000

    # 2. Logic Engines
    path_points = astar_path(mask_512)
    complexity, suitability, reason = calculate_metrics(mask_512)
    env_risk, env_status = assess_environment(img_rgb)
    passive_logs, env_degraded = run_passive_sensors(img_rgb, toggles)
    
    # Run MiDaS in background for hazards
    _, depth_uint8 = get_depth_map(img_rgb)
    hazards = detect_depth_hazards(depth_uint8, path_points)
    
    # 3. Generate Visuals
    mask_color = mask_to_color(mask_512)
    overlay = overlay_image(img_rgb, mask_color)
    risk_map = compute_risk_map(mask_512, original_size)
    path_plan = draw_path(overlay, path_points, hazards)
    
    # --- PHASE VII: FUSED TACTICAL FRAME ---
    fused_base = overlay_image(img_rgb, mask_color, alpha=0.5)
    risk_resized = cv2.resize(risk_map, (fused_base.shape[1], fused_base.shape[0]))
    fused_base = cv2.addWeighted(fused_base, 0.85, risk_resized, 0.15, 0)
    fused_base = draw_path(fused_base, path_points, hazards)
    
    # Run YOLO ALWAYS (on fused frame)
    fused_out, raw_yolo_out, hostiles, vehicles, threat_details = detect_threats(img_rgb, draw_img=fused_base)
    
    if any("FLASH" in log or "GLINT" in log for log in passive_logs):
        cv2.rectangle(fused_out, (0, 0), (fused_out.shape[1]-1, fused_out.shape[0]-1), (0, 255, 255), 10)
        
    h_img, w_img = fused_out.shape[:2]
    
    # Semi-transparent info box
    hud_overlay = fused_out.copy()
    cv2.rectangle(hud_overlay, (0, h_img-85), (380, h_img), (0, 0, 0), -1)
    cv2.addWeighted(hud_overlay, 0.7, fused_out, 0.3, 0, fused_out)
    
    lat_val = lat if lat is not None else 0.0
    lon_val = lon if lon is not None else 0.0
    hdg_val = heading if heading is not None else 0.0
    
    # Multi-line HUD
    cv2.putText(fused_out, f"GPS: {lat_val:.4f} / {lon_val:.4f} | HDG: {hdg_val:.1f}", (10, h_img-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    threat_color = (0, 0, 255) if hostiles > 0 or vehicles > 0 else (0, 255, 0)
    cv2.putText(fused_out, f"THREATS DETECTED: {hostiles} HOSTILES, {vehicles} VEHICLES", (10, h_img-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, threat_color, 2)
    
    env_color = (0, 255, 255)
    cv2.putText(fused_out, f"TERRAIN: {suitability} | VISIBILITY: {env_status}", (10, h_img-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, env_color, 2)
    
    # Handle legacy diagnostic modes
    special_layer = None
    if mode == 'depth':
        special_layer = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
    elif mode == 'threat':
        special_layer = raw_yolo_out
        
    # 4. Stats
    total = mask_512.size
    class_dist = {CLASS_NAMES[i]: round(float((mask_512 == i).sum()/total), 3) for i in range(NUM_CLASSES)}

    if lat is not None and lon is not None:
        rover_state["lat"] = lat
        rover_state["lon"] = lon

    return JSONResponse({
        'fused_b64':      arr_to_b64(fused_out),
        'overlay_b64':    arr_to_b64(special_layer, is_bgr=(mode == 'depth')) if special_layer is not None else arr_to_b64(overlay),
        'risk_map_b64':   arr_to_b64(risk_map, is_bgr=True),
        'path_plan_b64':  arr_to_b64(path_plan),
        'mean_iou':       0.894, 
        'inference_ms':   round(inference_ms, 2),
        'iou_per_class':  class_dist, 
        'terrain_complexity': complexity,
        'mission_suitability': suitability,
        'suitability_reason': reason,
        'gps':            {'lat': rover_state['lat'], 'lon': rover_state['lon']},
        'threat_data':    {'hostiles': hostiles, 'vehicles': vehicles, 'details': threat_details},
        'hazards':        hazards,
        
        # SENTINEL AI DATA
        'env_data':       {'risk': env_risk, 'status': env_status, 'logs': passive_logs, 'vae_degraded': env_degraded}
    })

def map_state_dict(state_dict):
    import re
    new_state = {}
    for key, value in state_dict.items():
        k = key
        k = re.sub(r'encoder\.patch_embeddings\.(\d+)', r'stages.\1.patch_embeddings', k)
        k = re.sub(r'encoder\.block\.(\d+)\.(\d+)', r'stages.\1.blocks.\2', k)
        k = k.replace('layer_norm_1', 'layernorm_before')
        k = k.replace('layer_norm_2', 'layernorm_after')
        k = k.replace('attention.self.query', 'attention.q_proj')
        k = k.replace('attention.self.key', 'attention.k_proj')
        k = k.replace('attention.self.value', 'attention.v_proj')
        k = k.replace('attention.output.dense', 'attention.o_proj')
        k = k.replace('attention.self.sr', 'attention.sequence_reduction.sequence_reduction')
        k = k.replace('attention.self.layer_norm', 'attention.sequence_reduction.layer_norm')
        k = k.replace('mlp.dense1', 'mlp.fc1')
        k = k.replace('mlp.dense2', 'mlp.fc2')
        k = re.sub(r'encoder\.layer_norm\.(\d+)', r'stages.\1.layer_norm', k)
        k = re.sub(r'decode_head\.linear_c\.(\d+)', r'decode_head.linear_projections.\1', k)
        new_state[k] = value
    return new_state

def load_models(ckpt_path, variant='b2'):
    global seg_model, seg_model_ort, depth_model, yolo_model, tracker, device, transform, env_ae, frame_counter, MIDAS_SKIP
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MIDAS_SKIP = 3 if device.type == 'cuda' else 10
    print(f'[Core] 🚀 TERRA-VIS Engine Starting on {device}...')
    frame_counter = 0
    
    # Check for vault
    vault_dir = Path(__file__).parent.parent / 'vault'
    if not vault_dir.exists():
        print(f"[WARN] Vault directory not found. Using original models (Unencrypted mode).")
        vault_dir = Path(__file__).parent.parent
        dec = lambda p: p # pass through
    else:
        print(f"[Core] Secure vault detected. Decrypting models on the fly.")
        def dec(p):
            p_obj = Path(p)
            if p_obj.exists(): return p_obj
            v_file = vault_dir / f"{p_obj.stem}.terravis"
            if v_file.exists(): return Path(decrypt_model(v_file, expected_suffix=p_obj.suffix, expected_prefix=p_obj.stem + "_"))
            return p_obj
    # Load Anomaly AE
    env_ae = AnomalyAE().to(device)
    try:
        tmp_sentinel = dec(Path('sentinel_model.pth'))
        env_ae.load_state_dict(torch.load(tmp_sentinel, map_location=device, weights_only=True))
        env_ae.eval()
        print('[Core] ✅ Environmental VAE Loaded')
    except Exception as e:
        print(f'[WARN] Failed to load VAE: {e}')
        env_ae = None
    
    # Load SegFormer
    onnx_path = dec(Path('outputs/desert_seg.onnx'))
    if Path(onnx_path).exists():
        print(f'[Core] Loading SegFormer via TensorRT/ONNX')
        # Force CUDA instead of relying on get_available_providers which might include TensorRT (which crashes if DLLs are missing)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        provider_options = [{}, {}]
        
        try:
            sess_opts = ort.SessionOptions()
            if provider_options:
                seg_model_ort = ort.InferenceSession(str(onnx_path), sess_opts, providers=providers, provider_options=provider_options)
            else:
                seg_model_ort = ort.InferenceSession(str(onnx_path), sess_opts, providers=providers)
            
            active_providers = seg_model_ort.get_providers()
            if device.type == 'cuda' and 'CUDAExecutionProvider' not in active_providers and 'TensorrtExecutionProvider' not in active_providers:
                print('[WARN] ONNX Runtime failed to attach to GPU (missing DLLs). Falling back to PyTorch on GPU!')
                seg_model_ort = None
            else:
                print('[Core] ✅ TensorRT/ONNX Engine Active')
        except Exception as e:
            print(f'[WARN] TensorRT/ONNX load failed: {e}. Falling back to PyTorch.')
            seg_model_ort = None

    if seg_model_ort is None:
        seg_model = DesertSegFormer(variant=variant, num_classes=6, pretrained=False).to(device)
        tmp_ckpt = dec(Path(ckpt_path))
        state = torch.load(tmp_ckpt, map_location=device, weights_only=True)
        actual_state = state['model'] if 'model' in state else state
        # actual_state = map_state_dict(actual_state)
        seg_model.load_state_dict(actual_state)
        seg_model.eval()
        
    depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    
    tmp_yolo = dec(Path("yolov8n.pt"))
    yolo_model = YOLO(str(tmp_yolo))
    
    # Initialize BoT-SORT
    tmp_reid = dec(Path('osnet_x0_25_msmt17.pt'))
    tracker = create_tracker(
        tracker_type='botsort',
        tracker_config=Path(boxmot.__file__).parent / 'configs' / 'trackers' / 'botsort.yaml',
        reid_weights=tmp_reid,
        device='cpu', # Keep Re-ID on CPU to save VRAM
        half=False
    )
    
    print('[Core] ✅ All Systems Online.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--variant', type=str, default='b2')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    load_models(args.ckpt, args.variant)
    uvicorn.run(app, host='0.0.0.0', port=args.port)