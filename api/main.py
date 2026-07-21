import sys
print("Python:", sys.executable)

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
    "Ham1mad1/videomae-base-Vsl-Lab-PC-V10"
)

classifier = pipeline(
    "video-classification",
    model="Ham1mad1/videomae-base-Vsl-Lab-PC-V10",
    image_processor=image_processor,
    device=device
)

@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    if not video.filename.endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(400, "Only video files allowed")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await video.read())
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