# RunPod Serverless "swap-only" endpoint using InsightFace inswapper_128
# Input JSON:
# {
#   "source_face_b64": "<base64>",          # required
#   "target_image_b64": "<base64>",         # required
#   "face_pick": "largest|first|all",       # optional (default "largest")
#   "min_confidence": 0.35                  # optional
# }
#
# Output:
# { "status": "success", "image_b64": "<base64>" }
# or
# { "status": "error", "message": "..." }

import base64
import os
import cv2
import numpy as np
import runpod
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# -------- Cold start: load once --------
DET_NAME = os.environ.get("IFACE_DET_NAME", "buffalo_l")
DET_SIZE = int(os.environ.get("IFACE_DET_SIZE", "640"))
SWAP_MODEL = os.environ.get("IFACE_SWAP_MODEL", "inswapper_128.onnx")  # auto-download
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "95"))

app = FaceAnalysis(name=DET_NAME)
app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))  # use GPU if available
swapper = get_model(SWAP_MODEL, download=True, download_zip=True)

def _b64_to_cv2(b64_str: str):
    data = base64.b64decode(b64_str)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode base64 image")
    return img

def _encode_jpg(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Failed to encode output image")
    return base64.b64encode(buf).decode("utf-8")

def _pick_faces(faces, face_pick: str):
    if not faces:
        return []
    face_pick = (face_pick or "largest").lower()
    if face_pick == "first":
        return [faces[0]]
    if face_pick == "all":
        return faces
    # default: largest bbox
    def area(f):
        x1, y1, x2, y2 = f.bbox.astype(int)
        return (x2 - x1) * (y2 - y1)
    return [max(faces, key=area)]

def handler(event):
    try:
        inp = event.get("input") or {}
        src_b64 = inp.get("source_face_b64")
        tgt_b64 = inp.get("target_image_b64")
        if not src_b64 or not tgt_b64:
            return {"status": "error", "message": "source_face_b64 and target_image_b64 are required."}

        face_pick = inp.get("face_pick", "largest")
        min_conf = float(inp.get("min_confidence", 0.35))

        src_img = _b64_to_cv2(src_b64)
        tgt_img = _b64_to_cv2(tgt_b64)

        src_faces = app.get(src_img)
        if not src_faces:
            return {"status": "error", "message": "No face detected in source image."}
        src_face = sorted(src_faces, key=lambda f: f.det_score, reverse=True)[0]
        if src_face.det_score < min_conf:
            return {"status": "error", "message": f"Low confidence for source face ({src_face.det_score:.2f} < {min_conf})."}

        tgt_faces = [f for f in app.get(tgt_img) if f.det_score >= min_conf]
        if not tgt_faces:
            return {"status": "error", "message": "No face detected in target image above min_confidence."}

        chosen = _pick_faces(tgt_faces, face_pick)
        out = tgt_img.copy()
        for f in chosen:
            out = swapper.get(out, f, src_face, paste_back=True)

        return {"status": "success", "image_b64": _encode_jpg(out)}

    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})
