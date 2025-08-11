import runpod
import cv2
import base64
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import numpy as np

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = get_model("inswapper_128.onnx", download=True, download_zip=True)

def handler(event):
    try:
        src_data = base64.b64decode(event["input"]["source"])
        tgt_data = base64.b64decode(event["input"]["target"])
        src_img = cv2.imdecode(np.frombuffer(src_data, np.uint8), cv2.IMREAD_COLOR)
        tgt_img = cv2.imdecode(np.frombuffer(tgt_data, np.uint8), cv2.IMREAD_COLOR)

        src_face = app.get(src_img)[0]
        tgt_faces = app.get(tgt_img)
        out = tgt_img.copy()
        for f in tgt_faces:
            out = swapper.get(out, f, src_face, paste_back=True)

        _, buffer = cv2.imencode(".jpg", out)
        result_b64 = base64.b64encode(buffer).decode("utf-8")
        return {"status": "success", "image": result_b64}
    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})
