from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

USER_AGENT = "Mozilla/5.0 (compatible; OpenCampusDemo/1.0)"
COLOR_MESH = (80, 200, 255)
COLOR_CONTOUR = (0, 220, 120)
COLOR_IRIS = (255, 180, 40)
COLOR_DOT = (255, 80, 80)


def download_bytes(url: str, save_path: Path, timeout: int = 120) -> Path:
    """URL からバイナリを保存する（既存ならスキップ）．

    Args:
        url (str): URL
        save_path (Path): 保存先
        timeout (int): 秒

    Returns:
        Path: 保存パス
    """
    if save_path.exists() and save_path.stat().st_size > 0:
        return save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        save_path.write_bytes(response.read())
    return save_path


def load_face_landmarker(model_path: Path, num_faces: int) -> vision.FaceLandmarker:
    """VIDEO モードの Face Landmarker を構築する．

    Args:
        model_path (Path): .task パス
        num_faces (int): 最大顔数

    Returns:
        vision.FaceLandmarker
    """
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=num_faces,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def landmark_to_pixel(landmark, width: int, height: int) -> tuple[int, int]:
    """正規化座標をピクセルへ変換する．

    Args:
        landmark: x,y 属性
        width (int): 幅
        height (int): 高さ

    Returns:
        tuple[int, int]: (x, y)
    """
    x = int(np.clip(landmark.x * width, 0, width - 1))
    y = int(np.clip(landmark.y * height, 0, height - 1))
    return x, y


def draw_connections(rgb, landmarks, connections, color, thickness=1) -> None:
    """接続線を描画する（インプレース）．

    Args:
        rgb (np.ndarray): RGB 画像
        landmarks: 1 顔分の点列
        connections: Connection リスト
        color (tuple[int, int, int]): RGB
        thickness (int): 線幅
    """
    h, w = rgb.shape[:2]
    for conn in connections:
        x1, y1 = landmark_to_pixel(landmarks[conn.start], w, h)
        x2, y2 = landmark_to_pixel(landmarks[conn.end], w, h)
        cv2.line(rgb, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)


model_path = Path(MODEL_PATH)
download_bytes(MODEL_URL, model_path)
face_landmarker = load_face_landmarker(model_path, MAX_NUM_FACES)
_frame_ts_ms = 0
print(f"モデル: {model_path} ({model_path.stat().st_size} bytes)")
print("初期化完了．次のセルでリアルタイム処理を開始してください．")
