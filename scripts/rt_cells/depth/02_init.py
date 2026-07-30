from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

MODEL_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def resolve_device() -> str:
    """利用可能な推論デバイスを返す．

    Returns:
        str: "cuda" または "cpu"
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {name} ({mem_gb:.1f} GB)")
        return "cuda"
    print("GPU が見つかりません．CPU で実行します．")
    return "cpu"


def resolve_colormap(name: str) -> int:
    """カラーマップ名を OpenCV 定数へ変換する．

    Args:
        name (str): 例 INFERNO

    Returns:
        int: cv2.COLORMAP_*
    """
    key = f"COLORMAP_{name.strip().upper()}"
    if not hasattr(cv2, key):
        return cv2.COLORMAP_INFERNO
    return int(getattr(cv2, key))


def load_depth_pipeline(model_size: str, device: str):
    """Depth Anything V2 パイプラインを構築する．

    Args:
        model_size (str): small/base/large
        device (str): cuda/cpu

    Returns:
        transformers Pipeline
    """
    model_id = MODEL_IDS[model_size.strip().lower()]
    device_index = 0 if device == "cuda" else -1
    print(f"モデル読込中: {model_id}")
    return pipeline(task="depth-estimation", model=model_id, device=device_index)


def depth_to_colormap(depth_gray: np.ndarray, colormap: int) -> np.ndarray:
    """深度を疑似カラー RGB にする．

    Args:
        depth_gray (np.ndarray): (H, W)
        colormap (int): cv2.COLORMAP_*

    Returns:
        np.ndarray: (H, W, 3) RGB
    """
    gray = np.asarray(depth_gray)
    if gray.ndim == 3:
        gray = gray[:, :, 0]
    gray = gray.astype(np.float32)
    d_min, d_max = float(gray.min()), float(gray.max())
    if d_max - d_min < 1e-6:
        norm = np.zeros_like(gray, dtype=np.uint8)
    else:
        norm = ((gray - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(norm, colormap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


DEVICE = resolve_device()
depth_pipe = load_depth_pipeline(MODEL_SIZE, DEVICE)
CMAP = resolve_colormap(COLORMAP_NAME)
print("初期化完了．次のセルでリアルタイム処理を開始してください．")
