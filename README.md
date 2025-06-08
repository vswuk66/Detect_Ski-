# Detect Ski

This project provides scripts for object tracking in jet ski videos and for converting YOLO models to ONNX format.

## Requirements

Install Python dependencies using:

```bash
pip install -r requirements.txt
```

## Converting a YOLO model

The `convert_model.py` script exports a YOLO model from the Ultralytics format (`.pt`) to ONNX.

```bash
python convert_model.py --model_path path/to/best.pt --onnx_path model.onnx
```

## Processing a video

The `process_video.py` script tracks a jet ski in a video and estimates its speed.

```bash
python process_video.py --video input.mp4 --output tracked.mp4 --model_path path/to/model.onnx
```

Press `q` while the window is focused to stop processing early.

