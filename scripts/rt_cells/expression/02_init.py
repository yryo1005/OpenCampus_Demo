from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

EMOTION_EN = {
    "angry": "Anger",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "sad": "Sad",
    "surprise": "Surprise",
    "neutral": "Neutral",
}
EMOTION_JA = {
    "angry": "怒り",
    "disgust": "嫌悪",
    "fear": "恐れ",
    "happy": "喜び",
    "sad": "悲しみ",
    "surprise": "驚き",
    "neutral": "無表情",
}


def resolve_device() -> str:
    """利用可能な推論デバイスを返す．

    Returns:
        str: cuda/cpu
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {name} ({mem_gb:.1f} GB)")
        return "cuda"
    print("GPU が見つかりません．CPU で実行します．")
    return "cpu"


def load_face_cascade() -> cv2.CascadeClassifier:
    """Haar Cascade 顔検出器を読み込む．

    Returns:
        cv2.CascadeClassifier
    """
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"cascade を読めません: {cascade_path}")
    return cascade


def load_emotion_pipeline(model_id: str, device: str):
    """表情分類パイプラインを構築する．

    Args:
        model_id (str): HF モデル ID
        device (str): cuda/cpu

    Returns:
        transformers Pipeline
    """
    device_index = 0 if device == "cuda" else -1
    print(f"モデル読込中: {model_id}")
    return pipeline(
        task="image-classification",
        model=model_id,
        device=device_index,
        top_k=7,
    )


def detect_largest_face(rgb, cascade, margin=0.25):
    """最大の顔枠を返す．

    Args:
        rgb (np.ndarray): (H,W,3)
        cascade: 検出器
        margin (float): 余白

    Returns:
        tuple[int,int,int,int] | None
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad_x, pad_y = int(w * margin), int(h * margin)
    h_img, w_img = rgb.shape[:2]
    return (
        max(0, x - pad_x),
        max(0, y - pad_y),
        min(w_img, x + w + pad_x),
        min(h_img, y + h + pad_y),
    )


device = resolve_device()
face_cascade = load_face_cascade()
emotion_clf = load_emotion_pipeline(MODEL_ID, device)
print("初期化完了．次のセルでリアルタイム処理を開始してください．")
