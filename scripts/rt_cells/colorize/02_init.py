from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import PyTorchModelHubMixin

DDCOLOR_DIR = Path("DDColor").resolve()
if not DDCOLOR_DIR.exists():
    raise FileNotFoundError("DDColor が見つかりません．セル1を先に実行してください．")
if str(DDCOLOR_DIR) not in sys.path:
    sys.path.insert(0, str(DDCOLOR_DIR))

from ddcolor import DDColor, ColorizationPipeline  # noqa: E402


class DDColorHF(DDColor, PyTorchModelHubMixin):
    """Hugging Face Hub 経由で重みを読み込む DDColor ラッパ．"""

    def __init__(self, config=None, **kwargs):
        if isinstance(config, dict):
            kwargs = {**config, **kwargs}
        super().__init__(**kwargs)


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


def load_colorizer(model_name: str, input_size: int, device: str) -> ColorizationPipeline:
    """DDColor を読み込み，着色パイプラインを返す．

    Args:
        model_name (str): HF リポジトリ名
        input_size (int): 入力一辺
        device (str): "cuda" / "cpu"

    Returns:
        ColorizationPipeline: BGR→BGR 着色器
    """
    repo_id = model_name if "/" in model_name else f"piddnad/{model_name}"
    print(f"モデル読込中: {repo_id}")
    model = DDColorHF.from_pretrained(repo_id)
    model = model.to(device)
    model.eval()
    print(f"読込完了: device={device}")
    return ColorizationPipeline(
        model,
        input_size=input_size,
        device=torch.device(device),
    )


def to_grayscale_rgb(rgb: np.ndarray) -> np.ndarray:
    """カラー RGB をモノクロ 3ch にする．

    Args:
        rgb (np.ndarray): (H, W, 3) uint8

    Returns:
        np.ndarray: モノクロ RGB
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def colorize_rgb(gray_rgb: np.ndarray, colorizer: ColorizationPipeline) -> np.ndarray:
    """モノクロ RGB を着色して RGB で返す．

    Args:
        gray_rgb (np.ndarray): (H, W, 3)
        colorizer (ColorizationPipeline): 着色器

    Returns:
        np.ndarray: 着色 RGB
    """
    bgr = cv2.cvtColor(gray_rgb, cv2.COLOR_RGB2BGR)
    out_bgr = colorizer.process(bgr)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


device = resolve_device()
colorizer = load_colorizer(MODEL_NAME, INPUT_SIZE, device)
print("初期化完了．次のセルでリアルタイム処理を開始してください．")
