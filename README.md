# 🏓 Table Tennis Stroke Classification API

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139-green)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13-orange)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/Adilmp/table-tennis-videomae)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)

REST API for classifying table tennis strokes from video with a fine-tuned **VideoMAE** (a Vision
Transformer for video). The model reaches **85.8% validation accuracy** across 21 classes: 20 strokes
plus a "Negative" (no stroke) class.

The model was trained in my final-year project at FAST-NUCES (2024); the training notebook is in
[Table-Tennis-Stroke-Classification-using-Advanced-Transformer-Architecture](https://github.com/Adilmp/Table-Tennis-Stroke-Classification-using-Advanced-Transformer-Architecture).

🔗 **Model:** [Hugging Face Hub](https://huggingface.co/Adilmp/table-tennis-videomae)
🔗 **API docs:** `http://localhost:8000/docs` (Swagger UI)

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Validation accuracy | **85.8%** (200 of 233 clips) |
| Classes | 21 (20 strokes + Negative) |
| Dataset | MediaEval table tennis strokes (Université de Bordeaux), 50 GB of video |
| Best earlier result on this dataset cited in our report | 68.78% (HCMUS, MediaEval 2021) |
| Architecture | VideoMAE (ViT-Base for video, 86M parameters) |
| CPU inference | about 1–2 s per clip |

---

## 🏗️ Architecture

```
Video upload → FastAPI → VideoMAE (Hugging Face Transformers) → JSON response
                                   ↓
                     21-class stroke classification
```

**Tech stack:** Python · PyTorch · VideoMAE · FastAPI · Transformers · Docker · ONNX Runtime

---

## 🚀 Quick Start

### Local development

```bash
# Clone
git clone https://github.com/Adilmp/table-tennis-stroke-api.git
cd table-tennis-stroke-api

# Set up (requirements.txt pulls the CPU build of PyTorch from PyTorch's own index)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs for interactive API documentation. The model (about 350 MB) is
downloaded from the Hugging Face Hub on first start.

### Docker

```bash
docker build -t tt-stroke-api .
docker run -p 8000:8000 tt-stroke-api
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Upload a video (`.mp4`, `.avi` or `.mov`) in the form field `video` → stroke label, confidence, inference time, top 5 |
| `/health` | GET | Server status, model name, device (CPU/CUDA) |

### Example request

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "video=@sample_video.mp4"
```

### Example response

```json
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
```

(Illustrative values; the fields are exactly what `/predict` returns.)

---

## 📁 Project Structure

```
table-tennis-stroke-api/
├── api/
│   └── main.py               # FastAPI application
├── scripts/
│   ├── export_onnx.py        # export the model to ONNX
│   └── onnx_inference.py     # run the ONNX model with ONNX Runtime (CPU)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🎯 Model Details

- **Base architecture:** VideoMAE (Masked Autoencoder for video): a Vision Transformer pre-trained by
  hiding most of each clip and reconstructing it, then fine-tuned for stroke classification
- **Training data:** the MediaEval table tennis stroke dataset from the Université de Bordeaux, 50 GB
  of player-centred videos recorded at a sports facility; 21 classes
- **Preprocessing:** 16 frames sampled per clip, normalisation, resizing to 224×224; augmentation
  (random resizing and cropping, horizontal flips) in earlier training phases
- **Fine-tuning:** learning rate 5e-5, batch size 10; about 49 epochs (4,000 steps), then 160 more
  steps from that checkpoint, keeping the best model by validation accuracy
- **Published** on the Hugging Face Hub for reproducible inference

---

## 🔮 Roadmap

- [x] ONNX export (`scripts/export_onnx.py`, `scripts/onnx_inference.py`)
- [ ] Tests for the API
- [ ] Batch video processing endpoint
- [ ] Real-time webcam inference
- [ ] Pose estimation integration (MediaPipe)

---

## ⚠️ Known Limitations

- **Camera angle sensitivity:** the training videos are player-centred clips from one sports
  facility; the model may generalise poorly to very different viewpoints (overhead, broadcast
  angles, behind the player).
- **Similar strokes:** loops are sometimes confused with hits, and serve spin types (sidespin,
  topspin, backspin) with each other.
- **Confidence isn't calibrated:** a wrong prediction can still come with a very high score.
- **Inference speed:** CPU inference takes about 1–2 s per clip; a GPU is recommended for real-time
  use.

---

## 📝 License

MIT
