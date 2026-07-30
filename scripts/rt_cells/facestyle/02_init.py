from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

HUB_REPO = "bryandlee/animegan2-pytorch:main"
STYLE_INFO = {
    "face_paint_512_v2": "アニメキャラ風",
    "face_paint_512_v1": "アニメ似顔絵風",
    "celeba_distill": "やさしいイラスト風",
    "paprika": "映画イラスト・絵画風",
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


def load_style_model(style_key: str, device: str) -> torch.nn.Module:
    """AnimeGANv2 Generator を1つ読み込む．

    Args:
        style_key (str): スタイル名
        device (str): cuda/cpu

    Returns:
        torch.nn.Module: eval 済みモデル
    """
    print(f"読み込み中: {style_key}")
    model = torch.hub.load(
        HUB_REPO,
        "generator",
        pretrained=style_key,
        device=device,
        progress=True,
        trust_repo=True,
    )
    model.eval()
    return model


def detect_largest_face(rgb, cascade, margin=0.45):
    """最大の顔枠を返す．

    Args:
        rgb (np.ndarray): (H,W,3)
        cascade: 検出器
        margin (float): 余白倍率

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


def crop_square_face(rgb, box):
    """顔または中央の正方形を切り出す．

    Args:
        rgb (np.ndarray): (H,W,3)
        box: (x1,y1,x2,y2) or None

    Returns:
        np.ndarray: 正方形 RGB
    """
    h, w = rgb.shape[:2]
    if box is not None:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        side = max(x2 - x1, y2 - y1)
    else:
        cx, cy = w // 2, h // 2
        side = min(h, w)
    half = side // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    x1 = max(0, x2 - side)
    y1 = max(0, y2 - side)
    return rgb[y1:y2, x1:x2]


def stylize_face(face_rgb, model, device, size):
    """顔をスタイル変換する．

    Args:
        face_rgb (np.ndarray): (H,W,3)
        model: Generator
        device (str): cuda/cpu
        size (int): 正方形解像度

    Returns:
        np.ndarray: 変換後 RGB (size,size,3)
    """
    pil = Image.fromarray(face_rgb).convert("RGB").resize((size, size), Image.LANCZOS)
    with torch.no_grad():
        x = to_tensor(pil).unsqueeze(0) * 2.0 - 1.0
        y = model(x.to(device)).cpu()[0]
        y = (y * 0.5 + 0.5).clamp(0.0, 1.0)
    return np.asarray(to_pil_image(y))


device = resolve_device()
face_cascade = load_face_cascade()
style_model = load_style_model(STYLE_KEY, device)
STYLE_JA = STYLE_INFO.get(STYLE_KEY, STYLE_KEY)
print(f"スタイル: {STYLE_JA}")
print("初期化完了．次のセルでリアルタイム処理を開始してください．")
