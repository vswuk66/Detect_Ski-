import os
import torch
import argparse
from ultralytics import YOLO
import tempfile


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Check or convert YOLO model to ONNX.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the YOLO model weights (.pt)."
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="Path to save the exported ONNX model."
    )
    return parser.parse_args()


def ensure_onnx_model(args):
    # Проверяем существование ONNX модели
    if os.path.isfile(args.onnx_path):
        print(f"ONNX model already exists: {args.onnx_path}")
        return

    print(f"ONNX model not found at {args.onnx_path}. Starting export...")
    os.environ['YOLO_CACHE'] = '0'
    os.environ['TMPDIR'] = tempfile.gettempdir()

    # Настраиваем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Загружаем YOLO модель
    model = YOLO(args.model_path)
    model.to(device)

    # Экспорт модели в ONNX
    print("Exporting YOLO model to ONNX format...")
    exported_path = model.export(
        format="onnx",
        imgsz=640,
        device=device,
        half=False)

    # Проверяем результат экспорта
    if exported_path and os.path.isfile(exported_path):
        print(f"Model successfully exported to ONNX: {exported_path}")
        if exported_path != args.onnx_path:
            os.rename(exported_path, args.onnx_path)
            print(f"Model renamed to: {args.onnx_path}")
    else:
        print("Failed to export YOLO model to ONNX.")
        exit(1)


if __name__ == "__main__":
    args = parse_arguments()
    ensure_onnx_model(args)
