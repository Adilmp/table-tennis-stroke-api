from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import pipeline, VideoMAEImageProcessor
import torch
import tempfile
import os
import time

app = FastAPI(title="Table Tennis Stroke Classifier")

device = 0 if torch.cuda.is_available() else -1

# Load image processor explicitly
image_processor = VideoMAEImageProcessor.from_pretrained(
    "Adilmp/table-tennis-videomae"
)

classifier = pipeline(
    "video-classification",
    model="Adilmp/table-tennis-videomae",
    image_processor=image_processor,
    device=device
)

# A plain def (not async): FastAPI runs it in a worker thread, so a slow prediction
# doesn't block other requests the way a synchronous model call inside async def would.
@app.post("/predict")
def predict(video: UploadFile = File(...)):
    if not video.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(400, "Only video files allowed")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video.file.read())
        tmp_path = tmp.name
    
    try:
        start = time.time()
        result = classifier(tmp_path)
        inference_time = (time.time() - start) * 1000
        
        return {
            "stroke": result[0]["label"],
            "confidence": round(result[0]["score"], 4),
            "inference_time_ms": round(inference_time, 2),
            "top_5": [
                {"label": r["label"], "score": round(r["score"], 4)} 
                for r in result[:5]
            ]
        }
    finally:
        os.unlink(tmp_path)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "videomae-base-Vsl-Lab-PC-V10",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }