# Table Tennis Stroke Classification API

Production-ready VideoMAE model for 21-class table tennis stroke classification.
Achieves **85.8% validation accuracy**.

## Quick Start

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
API Endpoints
Table
Endpoint	Method	Description
/predict	POST	Upload video file, returns stroke classification
/health	GET	Server and model status
Tech Stack
Python, PyTorch, VideoMAE, FastAPI, Transformers, Docker
