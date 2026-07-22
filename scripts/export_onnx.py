import torch
from transformers import VideoMAEForVideoClassification

MODEL_ID = "Adilmp/table-tennis-videomae"
ONNX_PATH = "table_tennis_videomae.onnx"

print(f"Loading model from {MODEL_ID}...")
model = VideoMAEForVideoClassification.from_pretrained(MODEL_ID)
model.eval()

# VideoMAE expects (batch, num_frames, channels, height, width)
# Your model config: num_frames=16, image_size=224
dummy_input = torch.randn(1, 16, 3, 224, 224)

print("Exporting to ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "logits": {0: "batch_size"}
    },
    opset_version=14,
    do_constant_folding=True
)
print(f"Success: {ONNX_PATH} created")
print(f"File size: {round(torch.os.path.getsize(ONNX_PATH) / 1024 / 1024, 2)} MB")
