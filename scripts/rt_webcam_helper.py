# Colab リアルタイム Webカメラ共通ヘルパ（ノートブック埋め込み用）
# このファイル自体は実行せず，generate_rt_notebooks.py が読み込む．

SOURCE = r'''
from __future__ import annotations

import json
import time
from base64 import b64decode, b64encode

import cv2
import numpy as np
from google.colab.output import eval_js
from IPython.display import Javascript, display

# ------------------------------------------------------------
# Colab 内リアルタイム Webカメラ（Gradio なし）
# ------------------------------------------------------------
_WEBCAM_JS = r"""
async function ocEnsureWebcam(width, height) {
  // セル再実行時は前回 UI / ストリームを破棄して作り直す
  if (window._ocVideo && window._ocVideo.srcObject) {
    window._ocVideo.srcObject.getTracks().forEach((t) => t.stop());
  }
  const old = document.getElementById('oc-rt-root');
  if (old) {
    old.remove();
  }
  window._ocReady = false;
  window._ocStop = false;

  const root = document.createElement('div');
  root.id = 'oc-rt-root';
  root.style.cssText = 'font-family:sans-serif;';

  const row = document.createElement('div');
  row.style.cssText = 'display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start;';

  const left = document.createElement('div');
  const right = document.createElement('div');
  left.innerHTML = '<div style="margin-bottom:4px;font-weight:600;">カメラ</div>';
  right.innerHTML = '<div style="margin-bottom:4px;font-weight:600;">AI処理結果</div>';

  const video = document.createElement('video');
  video.id = 'oc-video';
  video.autoplay = true;
  video.playsInline = true;
  video.muted = true;
  video.width = width;
  video.style.cssText = 'max-width:100%;background:#111;border-radius:8px;';

  const canvas = document.createElement('canvas');
  canvas.id = 'oc-canvas';
  canvas.style.display = 'none';

  const outImg = document.createElement('img');
  outImg.id = 'oc-out';
  outImg.width = width;
  outImg.style.cssText = 'max-width:100%;background:#111;border-radius:8px;min-height:120px;';

  const status = document.createElement('div');
  status.id = 'oc-status';
  status.style.cssText = 'margin:8px 0;color:#333;';
  status.textContent = 'カメラ起動中…';

  const stopBtn = document.createElement('button');
  stopBtn.id = 'oc-stop';
  stopBtn.textContent = '停止';
  stopBtn.style.cssText =
    'padding:8px 18px;font-size:16px;cursor:pointer;border-radius:6px;border:1px solid #888;background:#f5f5f5;';

  left.appendChild(video);
  right.appendChild(outImg);
  row.appendChild(left);
  row.appendChild(right);
  root.appendChild(row);
  root.appendChild(status);
  root.appendChild(stopBtn);
  document.body.appendChild(root);

  const stream = await navigator.mediaDevices.getUserMedia({
    video: {facingMode: 'user', width: {ideal: width}, height: {ideal: height}},
    audio: false,
  });
  video.srcObject = stream;
  await video.play();

  window._ocVideo = video;
  window._ocCanvas = canvas;
  window._ocOut = outImg;
  window._ocStatus = status;
  window._ocStop = false;
  stopBtn.onclick = () => {
    window._ocStop = true;
    status.textContent = '停止中…';
    stopBtn.disabled = true;
  };
  window._ocReady = true;
  status.textContent = '準備完了．Python 側のループが処理を開始します．';
}

async function ocCaptureFrame(mirror, quality) {
  const video = window._ocVideo;
  const canvas = window._ocCanvas;
  if (!video || video.readyState < 2) {
    return null;
  }
  const w = video.videoWidth || video.width;
  const h = video.videoHeight || Math.round(w * 0.75);
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (mirror) {
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(video, 0, 0, w, h);
  return canvas.toDataURL('image/jpeg', quality);
}

function ocUpdateResult(dataUrl) {
  if (window._ocOut) {
    window._ocOut.src = dataUrl;
  }
}

function ocSetStatus(text) {
  if (window._ocStatus) {
    window._ocStatus.textContent = text;
  }
}

function ocShouldStop() {
  return !!window._ocStop;
}

async function ocShutdown() {
  const video = window._ocVideo;
  if (video && video.srcObject) {
    video.srcObject.getTracks().forEach((t) => t.stop());
  }
  if (window._ocStatus) {
    window._ocStatus.textContent = '停止しました．再実行する場合はこのセルを再度実行してください．';
  }
}
"""


def rgb_to_data_url(rgb: np.ndarray, quality: int = 75) -> str:
    """RGB uint8 画像を JPEG data URL に変換する．

    Args:
        rgb (np.ndarray): RGB 画像，形状 (H, W, 3)，dtype=uint8
        quality (int): JPEG 品質（1–100）

    Returns:
        str: data:image/jpeg;base64,... 形式の文字列
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG エンコードに失敗しました")
    return "data:image/jpeg;base64," + b64encode(buf.tobytes()).decode("ascii")


def data_url_to_rgb(data_url: str | None) -> np.ndarray | None:
    """JPEG data URL を RGB uint8 画像へ変換する．

    Args:
        data_url (str | None): data:image/jpeg;base64,... または None

    Returns:
        np.ndarray | None: RGB 画像 (H, W, 3)．失敗時は None
    """
    if not data_url or "," not in data_url:
        return None
    raw = b64decode(data_url.split(",", 1)[1])
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_long_side_rt(rgb: np.ndarray, max_side: int) -> np.ndarray:
    """長辺が max_side を超える場合に縮小する．

    Args:
        rgb (np.ndarray): RGB 画像，形状 (H, W, 3)
        max_side (int): 長辺の上限

    Returns:
        np.ndarray: 必要なら縮小した RGB 画像
    """
    h, w = rgb.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return rgb
    scale = max_side / float(long_side)
    return cv2.resize(
        rgb,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def draw_fps_label(rgb: np.ndarray, fps: float, extra: str = "") -> np.ndarray:
    """左上に FPS と補足テキストを描画する．

    Args:
        rgb (np.ndarray): RGB 画像，形状 (H, W, 3)
        fps (float): 表示する FPS
        extra (str): 追加テキスト

    Returns:
        np.ndarray: 描画後の RGB 画像
    """
    out = rgb.copy()
    text = f"{fps:.1f} FPS"
    if extra:
        text = f"{text} | {extra}"
    cv2.rectangle(out, (8, 8), (8 + 12 * len(text) + 24, 40), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def make_side_by_side_rt(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """2枚の RGB 画像を同じ高さで横並びにする．

    Args:
        left (np.ndarray): 左画像 (H, W, 3)
        right (np.ndarray): 右画像 (H, W, 3)

    Returns:
        np.ndarray: 横連結画像
    """
    h = min(left.shape[0], right.shape[0])
    w = min(left.shape[1], right.shape[1])
    l = cv2.resize(left, (w, h), interpolation=cv2.INTER_AREA)
    r = cv2.resize(right, (w, h), interpolation=cv2.INTER_AREA)
    return np.concatenate([l, r], axis=1)


def start_realtime_webcam(width: int = 640, height: int = 480) -> None:
    """Colab 上に Webカメラ UI を表示する．

    Args:
        width (int): 表示幅の目安
        height (int): getUserMedia の ideal 高さ
    """
    display(Javascript(_WEBCAM_JS))
    eval_js(f"ocEnsureWebcam({int(width)}, {int(height)})")
    print("カメラ UI を表示しました．ブラウザのカメラ許可をオンにしてください．")


def run_realtime_loop(
    process_fn,
    *,
    mirror: bool = True,
    max_side: int = 512,
    jpeg_quality: float = 0.7,
    max_frames: int | None = None,
) -> None:
    """Webカメラ映像を連続取得し，process_fn で処理して結果を更新する．

    Args:
        process_fn: RGB (H,W,3) を受け取り RGB (H,W,3) を返す関数
        mirror (bool): カメラ映像を左右反転するか（自撮り表示）
        max_side (int): 推論前の長辺上限
        jpeg_quality (float): キャプチャ JPEG 品質（0–1）
        max_frames (int | None): 最大フレーム数（None で停止ボタンまで）
    """
    start_realtime_webcam(width=640, height=480)
    time.sleep(0.4)

    n = 0
    t0 = time.perf_counter()
    fps = 0.0
    while True:
        if eval_js("ocShouldStop()"):
            break
        if max_frames is not None and n >= max_frames:
            break

        data_url = eval_js(
            f"ocCaptureFrame({str(bool(mirror)).lower()}, {float(jpeg_quality)})"
        )
        rgb = data_url_to_rgb(data_url)
        if rgb is None:
            time.sleep(0.05)
            continue

        rgb = resize_long_side_rt(rgb, int(max_side))
        t_inf0 = time.perf_counter()
        try:
            out = process_fn(rgb)
        except Exception as exc:  # noqa: BLE001
            eval_js(f"ocSetStatus({json.dumps('エラー: ' + str(exc))})")
            time.sleep(0.2)
            continue
        dt = max(time.perf_counter() - t_inf0, 1e-6)
        inst_fps = 1.0 / dt
        n += 1
        elapsed = time.perf_counter() - t0
        fps = n / max(elapsed, 1e-6)
        if out is None:
            continue
        vis = draw_fps_label(out, fps, extra=f"infer {inst_fps:.1f}")
        eval_js(f"ocUpdateResult({json.dumps(rgb_to_data_url(vis))})")
        eval_js(
            f"ocSetStatus({json.dumps(f'稼働中: {n} frames / 平均 {fps:.1f} FPS（停止ボタンで終了）')})"
        )

    eval_js("ocShutdown()")
    print(f"ループ終了: {n} frames, 平均 {fps:.1f} FPS")
'''.lstrip("\n")
