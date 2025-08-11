import base64, os
import cv2, numpy as np
import runpod
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

DET_NAME = os.environ.get("IFACE_DET_NAME", "buffalo_l")
DET_SIZE = int(os.environ.get("IFACE_DET_SIZE", "640"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "95"))
INSIGHTFACE_HOME = os.environ.get("INSIGHTFACE_HOME", "/app/.insightface")

def _load_detector():
    os.makedirs(INSIGHTFACE_HOME, exist_ok=True)
    app = FaceAnalysis(name=DET_NAME, root=INSIGHTFACE_HOME)
    app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))  # GPU if present, else CPU
    return app

def _load_swapper():
    local = os.path.join(INSIGHTFACE_HOME, "models", "inswapper_128.onnx")
    if os.path.exists(local):
        return get_model(local, download=False)
    return get_model("inswapper_128.onnx", download=True, download_zip=True)

app = _load_detector()
swapper = _load_swapper()

def _b64_to_cv2(b64_str: str):
    arr = np.frombuffer(base64.b64decode(b64_str), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode base64 image")
    return img

def _encode_jpg(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Failed to encode output image")
    return base64.b64encode(buf).decode("utf-8")

def _pick_faces(faces, mode: str):
    if not faces: return []
    mode = (mode or "largest").lower()
    if mode == "first": return [faces[0]]
    if mode == "all": return faces
    def area(f): x1,y1,x2,y2 = f.bbox.astype(int); return (x2-x1)*(y2-y1)
    return [max(faces, key=area)]

def handler(event):
    try:
        inp = event.get("input") or {}
        src_b64 = inp.get("source_face_b64")
        tgt_b64 = inp.get("target_image_b64")
        if not src_b64 or not tgt_b64:
            return {"status":"error","message":"source_face_b64 and target_image_b64 are required."}

        face_pick = inp.get("face_pick","largest")
        min_conf  = float(inp.get("min_confidence", 0.35))

        src_img = _b64_to_cv2(src_b64)
        tgt_img = _b64_to_cv2(tgt_b64)

        src_faces = app.get(src_img)
        if not src_faces:
            return {"status":"error","message":"No face detected in source image."}
        src_face = max(src_faces, key=lambda f: f.det_score)
        if src_face.det_score < min_conf:
            return {"status":"error","message":f"Low confidence for source face ({src_face.det_score:.2f} < {min_conf})."}

        tgt_faces = [f for f in app.get(tgt_img) if f.det_score >= min_conf]
        if not tgt_faces:
            return {"status":"error","message":"No face detected in target image above min_confidence."}

        out = tgt_img.copy()
        for f in _pick_faces(tgt_faces, face_pick):
            out = swapper.get(out, f, src_face, paste_back=True)

        return {"status":"success","image_b64": _encode_jpg(out)}
    except Exception as e:
        return {"status":"error","message": str(e)}

runpod.serverless.start({"handler": handler})
