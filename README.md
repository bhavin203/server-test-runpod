# RunPod Serverless — Face Swap (InsightFace)

Serverless endpoint that swaps a source face into a target image using `insightface` (`buffalo_l` detector + `inswapper_128.onnx`). Optimized for T4 GPUs and fallback usage.

## Deploy
1. RunPod → **Serverless** → **Create Endpoint** (Python 3.10 GPU base).
2. Upload `handler.py` and `requirements.txt`.
3. (Optional env):
   - `IFACE_DET_NAME=buffalo_l`
   - `IFACE_DET_SIZE=640`
   - `IFACE_SWAP_MODEL=inswapper_128.onnx`
   - `JPEG_QUALITY=95`
4. Deploy → copy your `ENDPOINT_ID`.

## Sync request
```bash
curl "https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "source_face_b64": "<base64_source>",
      "target_image_b64": "<base64_target>",
      "face_pick": "largest",
      "min_confidence": 0.35
    }
  }'
