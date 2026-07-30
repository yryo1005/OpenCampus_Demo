from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

CHECKPOINT_DIR = Path("models")
CHECKPOINT_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
}
CHECKPOINT_FILENAMES = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}
USER_AGENT = "Mozilla/5.0 (compatible; OpenCampusDemo/1.0)"


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


def download_bytes(url: str, save_path: Path) -> Path:
    """チェックポイント等をダウンロードする．

    Args:
        url (str): URL
        save_path (Path): 保存先

    Returns:
        Path
    """
    if save_path.exists() and save_path.stat().st_size > 0:
        return save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=600) as response:
        save_path.write_bytes(response.read())
    return save_path


def load_sam_predictor(model_type: str, device: str) -> SamPredictor:
    """SAM Predictor を構築する．

    Args:
        model_type (str): vit_b 等
        device (str): cuda/cpu

    Returns:
        SamPredictor
    """
    ckpt_path = CHECKPOINT_DIR / CHECKPOINT_FILENAMES[model_type]
    print(f"チェックポイント準備: {ckpt_path.name}")
    download_bytes(CHECKPOINT_URLS[model_type], ckpt_path)
    print(f"  サイズ: {ckpt_path.stat().st_size / (1024**2):.1f} MB")
    sam = sam_model_registry[model_type](checkpoint=str(ckpt_path))
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """マスクを半透明色で重ねる．

    Args:
        rgb (np.ndarray): (H,W,3)
        mask (np.ndarray): (H,W) bool/0-1
        alpha (float): 透過度

    Returns:
        np.ndarray: オーバーレイ RGB
    """
    out = rgb.copy()
    m = mask.astype(bool)
    color = np.array([30, 180, 255], dtype=np.float32)
    out[m] = (out[m].astype(np.float32) * (1 - alpha) + color * alpha).astype(np.uint8)
    return out


DEVICE = resolve_device()
predictor = load_sam_predictor(SAM_MODEL_TYPE, DEVICE)
_sam_frame_i = 0
_last_mask = None
print("初期化完了．次のセルでリアルタイム処理を開始してください．")
