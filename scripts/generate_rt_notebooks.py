#!/usr/bin/env python3
"""scripts/rt_cells と rt_webcam_helper から OC_*_RT.ipynb を組み立てる．"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = Path(__file__).resolve().parent / "rt_cells"

# import helper SOURCE
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rt_webcam_helper import SOURCE as WEBCAM_HELPER  # noqa: E402


def md(text: str) -> dict:
    text = text.strip("\n")
    lines = text.split("\n")
    src = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines else [])
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(text: str) -> dict:
    text = text.strip("\n")
    if text and not text.endswith("\n"):
        text += "\n"
    lines = text.split("\n")
    # keep trailing newline as jupyter style: all lines end with \n except possibly last empty
    if lines and lines[-1] == "":
        lines = lines[:-1]
    src = [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else [])
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def badge(nb: str) -> str:
    return (
        "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"(https://colab.research.google.com/github/yryo1005/OpenCampus_Demo/blob/main/{nb})"
    )


def write_nb(name: str, cells: list[dict]) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "colab": {"provenance": [], "gpuType": "T4"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }
    out = ROOT / name
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(cells)} cells)")


def standard_flow(
    *,
    nb_name: str,
    title_md: str,
    settings_note: str,
    folder: str,
) -> None:
    d = CELLS / folder
    cells = [
        md(badge(nb_name)),
        md(title_md),
        md("## 0. 設定\n\n" + settings_note),
        code(read(d / "00_settings.py")),
        md("## 1. ライブラリのインストール"),
        code(read(d / "01_install.py")),
        md(
            "## 2. ライブラリの読み込み・初期化\n\n"
            "モデルを読み込みます．初回はダウンロードに時間がかかることがあります．"
        ),
        code(read(d / "02_init.py")),
        md(
            "## 3. リアルタイム処理（Gradio なし）\n\n"
            "セルを実行するとカメラ映像と処理結果が並んで表示されます．"
            "**停止**ボタンで終了します．"
        ),
        code(WEBCAM_HELPER + "\n" + read(d / "03_run_tail.py")),
    ]
    write_nb(nb_name, cells)


def main() -> None:
    standard_flow(
        nb_name="OC_Colorize_RT.ipynb",
        title_md="""# AI着色デモ（リアルタイム）

顔の Webカメラ映像をいったん **モノクロ** にし，AI（[DDColor](https://github.com/piddnad/DDColor)）で着色して **リアルタイム表示** します．

- **Gradio は使いません**（Colab 内の JavaScript + Python ループ）
- API キー / Hugging Face 認証は **不要**
- 推奨ランタイム: **T4 GPU**
""",
        settings_note=(
            "- カメラが左右逆なら `MIRROR_WEBCAM` を切り替え\n"
            "- リアルタイム向けに軽量モデル・小さめ解像度を既定にしています"
        ),
        folder="colorize",
    )

    standard_flow(
        nb_name="OC_DepthAnything_RT.ipynb",
        title_md="""# 深度推定デモ（リアルタイム）

Webカメラ映像から「手前／奥」を推定し，カラーマップで **リアルタイム可視化** します（Depth Anything V2）．

- **Gradio は使いません**
- API キー / Hugging Face 認証は **不要**
- 推奨ランタイム: **T4 GPU**
""",
        settings_note="- `MODEL_SIZE=small` が T4 向け推奨です",
        folder="depth",
    )

    standard_flow(
        nb_name="OC_FaceLandmark_RT.ipynb",
        title_md="""# 顔ランドマーク検出デモ（リアルタイム）

Webカメラの顔から特徴点（最大 478 点）を検出し，メッシュを **リアルタイム描画** します（MediaPipe）．

- **Gradio は使いません**
- API キー不要．CPU でも動作可（GPU 推奨）
""",
        settings_note="- メッシュが見づらいときは `DRAW_TESSELATION` などを切り替え",
        folder="landmark",
    )

    standard_flow(
        nb_name="OC_FaceStyle_RT.ipynb",
        title_md="""# 顔スタイル変換デモ（リアルタイム）

Webカメラの顔を AnimeGANv2 でアニメ／イラスト風に **リアルタイム変換** します．

- **Gradio は使いません**
- API キー不要（`torch.hub` で重み取得）
- 推奨ランタイム: **T4 GPU**
""",
        settings_note="- `STYLE_KEY` でスタイルを変更（変更後はセル2以降を再実行）",
        folder="facestyle",
    )

    standard_flow(
        nb_name="OC_FacialExpression_RT.ipynb",
        title_md="""# 表情認識デモ（リアルタイム）

Webカメラの顔から 7 種類の表情を推定し，枠とラベルを **リアルタイム表示** します．

- **Gradio は使いません**
- API キー / Hugging Face 認証は **不要**
- 推奨ランタイム: **T4 GPU**
""",
        settings_note="- 表情クラス: 喜び・悲しみ・怒り・驚き・恐れ・嫌悪・無表情",
        folder="expression",
    )

    d = CELLS / "sam"
    write_nb(
        "OC_SegmentAnything_RT.ipynb",
        [
            md(badge("OC_SegmentAnything_RT.ipynb")),
            md(
                """# セグメンテーションデモ（リアルタイム）

Webカメラ映像の中央付近の物体を SAM で切り出し，色付きオーバーレイを **連続表示** します．

- **Gradio は使いません**
- リアルタイム向けに **中央点プロンプト**（自動全分割より高速）
- API キー不要．推奨ランタイム: **T4 GPU**
"""
            ),
            md(
                "## 0. 設定\n\n"
                "- 画面中央の十字がプロンプト点です（顔や手元を中央に置くと分かりやすい）\n"
                "- `EMBED_EVERY_N` を大きくすると速いが追従が遅れます"
            ),
            code(read(d / "00_settings.py")),
            md("## 1. ライブラリのインストール"),
            code(read(d / "01_install.py")),
            md(
                "## 2. ライブラリの読み込み・初期化\n\n"
                "モデルを読み込みます．初回はダウンロードに時間がかかることがあります．"
            ),
            code(read(d / "02_init.py")),
            md(
                "## 3. リアルタイム処理（Gradio なし）\n\n"
                "セルを実行するとカメラ映像とセグメンテーション結果が表示されます．"
                "中央の十字の物体が色分けされます．**停止**ボタンで終了します．"
            ),
            code(WEBCAM_HELPER + "\n" + read(d / "03_run_tail.py")),
        ],
    )


if __name__ == "__main__":
    main()
