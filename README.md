# 🏓 Table Tennis Stroke Classification API

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Adilmp/table-tennis-videomae)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)

Production-ready REST API for classifying table tennis strokes from video using a fine-tuned **VideoMAE (Vision Transformer)** model. Achieves **85.8% validation accuracy** across 21 stroke categories.

🔗 **Live Model:** [Hugging Face Hub](https://huggingface.co/Adilmp/table-tennis-videomae)  
🔗 **API Docs:** `http://localhost:8000/docs` (Swagger UI)

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | **85.8%** |
| Classes | 21 |
| Dataset Size | 50GB sports video |
| Baseline (CNN) | ~72% |
| Architecture | VideoMAE (ViT for video) |

---

## 🏗️ Architecture
Video Upload → FastAPI → VideoMAE Inference → JSON Response
↓
Hugging Face Transformers
↓
21-class Stroke Classification

**Tech Stack:** Python · PyTorch · VideoMAE · FastAPI · Transformers · Docker

---

## 🚀 Quick Start

### Local Development

```bash
# Clone
git clone https://github.com/Adilmp/table-tennis-stroke-api.git
cd table-tennis-stroke-api

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
Open http://localhost:8000/docs for interactive API documentation.
Docker
bash
docker build -t tt-stroke-api .
docker run -p 8000:8000 tt-stroke-api
📡 API Endpoints
Table
Endpoint	Method	Description
/predict	POST	Upload .mp4 video → returns stroke label, confidence, inference time
/health	GET	Server status, model info, device (CPU/CUDA)
Example Request
bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@sample_video.mp4"
Example Response
JSON
{
  "stroke": "Offensive Forehand Loop",
  "confidence": 0.8523,
  "inference_time_ms": 1245.67,
  "top_5": [
    {"label": "Offensive Forehand Loop", "score": 0.8523},
    {"label": "Offensive Forehand Hit", "score": 0.0891},
    {"label": "Offensive Backhand Loop", "score": 0.0342},
    {"label": "Serve Forehand Loop", "score": 0.0124},
    {"label": "Negative", "score": 0.0089}
  ]
}
📁 Project Structure
plain
table-tennis-stroke-api/
├── api/
│   └── main.py              # FastAPI application
├── notebooks/
│   └── training_videomae.ipynb   # Original training pipeline
├── scripts/
│   └── export_onnx.py       # ONNX optimization (WIP)
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
🎯 Model Details
Base Architecture: VideoMAE (Masked Autoencoder for Video) — Vision Transformer adapted for temporal video understanding
Training Data: 50GB of table tennis match footage
Preprocessing: Frame sampling (16 frames), normalization, resizing to 224×224
Fine-tuning: 2 epochs, learning rate 5e-5, batch size 10
Export: Available on Hugging Face Hub for reproducible inference

🔮 Roadmap
[ ] ONNX export for optimized inference latency
[ ] Batch video processing endpoint
[ ] Real-time webcam inference
[ ] Pose estimation integration (MediaPipe)
📝 License
MIT
