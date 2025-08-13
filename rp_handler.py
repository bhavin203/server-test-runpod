import os, sys, io, zipfile, base64, traceback, glob, shutil
import requests
import numpy as np
import cv2
import runpod

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# ----------------- Config via env -----------------
DET_NAME = os.environ.get("IFACE_DET_NAME", "buffalo_l")
DET_SIZE = int(os.environ.get("IFACE_DET_SIZE", "640"))
CTX_ID   = int(os.environ.get("IFACE_CTX_ID", "-1"))   # -1 CPU, 0 GPU0 (if GPU endpoint later)
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "95"))
INSIGHTFACE_HOME = os.environ.get("INSIGHTFACE_HOME", "/app/.insightface")

BUFFALO_ZIP_URL     = os.environ.get("BUFFALO_ZIP_URL")
BUFFALO_ZIP_URL_ALT = os.environ.get("BUFFALO_ZIP_URL_ALT")
INSWAPPER_URL       = os.environ.get("INSWAPPER_URL")
INSWAPPER_URL_ALT   = os.environ.get("INSWAPPER_URL_ALT")

# Required files inside buffalo_l
BUFFALO_FILES = ["det_10g.onnx", "w600k_r50.onnx"]  # (others may be present; these two are essential)

app = None
swapper = None

# ----------------- Utils -----------------
def log(msg): print(str(msg), flush=True)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def download_with_fallback(urls, dst_path, stream=False):
    last_err = None
    for url in urls:
        try:
            log(f"Downloading: {url} -> {dst_path}")
            with requests.get(url, stream=stream, timeout=300, allow_redirects=True) as r:
                r.raise_for_status()
                if stream:
                    with open(dst_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1<<20):
                            if chunk:
                                f.write(chunk)
                else:
                    open(dst_path, "wb").write(r.content)
            return dst_path
        except Exception as e:
            last_err = e
            log(f"Download failed from {url}: {e}")
    raise RuntimeError(f"All mirrors failed for {dst_path}: {last_err}")

def flatten_into(target_dir):
    # Move all .onnx files found under target_dir/** to target_dir root
    for f in glob.glob(os.path.join(target_dir, "**", "*.onnx"), recursive=True):
        base = os.path.basename(f)
        dst = os.path.join(target_dir, base)
        if os.path.abspath(f) != os.path.abspath(dst):
            try:
                shutil.move(f, dst)
            except Exception:
                pass

# ----------------- Model Ensurers -----------------
def ensure_buffalo():
    """Ensure buffalo_l models exist locally so InsightFace doesn't try remote."""
    models_root = ensure_dir(os.path.join(INSIGHTFACE_HOME, "models"))
    buffalo_dir = ensure_dir(os.path.join(models_root, "buffalo_l"))

    # If the two key files exist, we're good.
    if all(os.path.exists(os.path.join(buffalo_dir, f)) for f in BUFFALO_FILES):
        return buffalo_dir

    # Otherwise, download ZIP from mirror(s) and extract.
    urls = [u for u in [BUFFALO_ZIP_URL, BUFFALO_ZIP_URL_ALT] if u]
    if not urls:
        raise RuntimeError("BUFFALO_ZIP_URL/BUFFALO_ZIP_URL_ALT not set and buffalo_l not present.")

    zip_bytes = io.BytesIO()
    # Some mirrors require streaming; we'll just stream to memory
    for url in urls:
        try:
            log(f"Fetching buffalo_l zip from {url}")
            with requests.get(url, timeout=300, allow_redirects=True) as r:
                r.raise_for_status()
                zip_bytes = io.BytesIO(r.content)
                break
        except Exception as e:
            log(f"Buffalo mirror failed: {e}")
            continue
    else:
        raise RuntimeError("Failed to fetch buffalo_l from all mirrors.")

    try:
        with zipfile.ZipFile(zip_bytes) as zf:
            zf.extractall(buffalo_dir)
    except zipfile.BadZipFile:
        # Some mirrors serve HTML unless 'download' query is correct. Try writing to disk then re-open.
        tmp_zip = os.path.join(models_root, "buffalo_l.tmp.zip")
        download_with_fallback(urls, tmp_zip, stream=True)
        with zipfile.ZipFile(tmp_zip) as zf:
            zf.extractall(buffalo_dir)
        os.remove(tmp_zip)

    # Flatten nested folders so files live directly under .../buffalo_l/
    flatten_into(buffalo_dir)

    # Validate files
    missing = [f for f in BUFFALO_FILES if not os.path.exists(os.path.join(buffalo_dir, f))]
    if missing:
        raise RuntimeError(f"buffalo_l missing required files after extract: {missing}")

    log("buffalo_l is ready.")
    return buffalo_dir

def ensure_inswapper():
    """Ensure inswapper_128.onnx exists locally."""
    models_root = ensure_dir(os.path.join(INSIGHTFACE_HOME, "models"))
    onnx_path = os.path.join(models_root, "inswapper_128.onnx")
    if os.path.exists(onnx_path):
        return onnx_path

    urls = [u for u in [INSWAPPER_URL, INSWAPPER_URL_ALT] if u]
    if not urls:
        raise RuntimeError("INSWAPPER_URL/ALT not set and inswapper_128.onnx not found.")
    download_with_fallback(urls, onnx_path, stream=True)
    log("inswapper_128.onnx is ready.")
    return onnx_path

def load_models():
    global app, swapper
    ensure_dir(INSIGHTFACE_HOME)
    buffalo_dir = ensure_buffalo()
    inswapper_path = ensure_inswapper()

    if app is None:
        log(f"Loading FaceAnalysis '{DET_NAME}' from {buffalo_dir} (ctx_id={CTX_ID})")
        app = FaceAnalysis(name=DET_NAME, root=INSIGHTFACE_HOME)
        app.prepare(ctx_id=CTX_ID, det_size=(DET_SIZE, DET_SIZE))
    if swapper is None:
        log("Loading inswapper_128 from local path")
        swapper = get_model(inswapper_path, download=False)
    return app, swapper

# ----------------- Helpers -----------------
def b64_to_cv2(b64_str: str):
    arr = np.frombuffer(base64.b64decode(b64_str), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode base64 image")
    return img

def encode_jpg(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Failed to encode output image")
    return base64.b64encode(buf).decode("utf-8")

def pick_faces(faces, mode: str):
    if not faces: return []
    mode = (mode or "largest").lower()
    if mode == "first": return [faces[0]]
    if mode == "all": return faces
    def area(f): x1,y1,x2,y2 = f.bbox.astype(int); return (x2-x1)*(y2-y1)
    return [max(faces, key=area)]

# ----------------- Handler -----------------
def handler(event):
    try:
        inp = (event or {}).get("input") or {}

        # warmup path to pre-load models
        if inp.get("warmup"):
            load_models()
            return {"status": "success", "message": "warmed"}

        src_b64 = inp.get("source_face_b64")
        tgt_b64 = inp.get("target_image_b64")
        if not src_b64 or not tgt_b64:
            return {"status": "error", "message": "source_face_b64 and target_image_b64 are required."}

        face_pick = inp.get("face_pick", "largest")
        min_conf  = float(inp.get("min_confidence", 0.35))

        app, swapper = load_models()

        src_img = b64_to_cv2(src_b64)
        tgt_img = b64_to_cv2(tgt_b64)

        src_faces = app.get(src_img)
        if not src_faces:
            return {"status": "error", "message": "No face detected in source image."}
        src_face = max(src_faces, key=lambda f: f.det_score)
        if src_face.det_score < min_conf:
            return {"status": "error", "message": f"Low confidence for source face ({src_face.det_score:.2f} < {min_conf})."}

        tgt_faces = [f for f in app.get(tgt_img) if f.det_score >= min_conf]
        if not tgt_faces:
            return {"status": "error", "message": "No face detected in target image above min_confidence."}

        out = tgt_img.copy()
        for f in pick_faces(tgt_faces, face_pick):
            out = swapper.get(out, f, src_face, paste_back=True)

        return {"status": "success", "image_b64": encode_jpg(out)}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})
