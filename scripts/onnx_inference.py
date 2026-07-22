import numpy as np
import onnxruntime as ort
from transformers import VideoMAEImageProcessor
import cv2
import time

MODEL_ID = "Adilmp/table-tennis-videomae"
ONNX_PATH = "table_tennis_videomae.onnx"

# Load processor (same as before)
processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)

# Load ONNX session
session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

# Load label mapping
id2label = processor.id2label if hasattr(processor, "id2label") else {
    0: "Defensive Backhand Backspin",
    1: "Defensive Backhand Block",
    # ... (full list from your notebook)
}

def preprocess_video(video_path: str):
    """Extract 16 frames and preprocess for VideoMAE."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, 16, dtype=int)
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    
    # Pad if video is too short
    while len(frames) < 16:
        frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    
    # Use HF processor
    inputs = processor(frames, return_tensors="np")
    return inputs["pixel_values"]

def predict(video_path: str):
    inputs = preprocess_video(video_path)
    start = time.time()
    outputs = session.run(None, {"pixel_values": inputs})
    latency = (time.time() - start) * 1000
    
    logits = outputs[0][0]
    probs = np.exp(logits) / np.sum(np.exp(logits))
    top_idx = np.argsort(probs)[::-1][:5]
    
    return {
        "stroke": id2label.get(int(top_idx[0]), f"class_{top_idx[0]}"),
        "confidence": round(float(probs[top_idx[0]]), 4),
        "inference_time_ms": round(latency, 2),
        "top_5": [
            {"label": id2label.get(int(i), f"class_{i}"), "score": round(float(probs[i]), 4)}
            for i in top_idx
        ]
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python onnx_inference.py path/to/video.mp4")
        exit(1)
    print(predict(sys.argv[1]))
