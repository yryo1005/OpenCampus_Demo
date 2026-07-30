def process_frame(rgb: np.ndarray) -> np.ndarray:
    """1フレームをモノクロ化→着色し，比較画像を返す．

    Args:
        rgb (np.ndarray): カメラ RGB (H, W, 3)

    Returns:
        np.ndarray: モノクロ|着色 の横並び
    """
    gray = to_grayscale_rgb(rgb)
    color = colorize_rgb(gray, colorizer)
    return make_side_by_side_rt(gray, color)


run_realtime_loop(
    process_frame,
    mirror=MIRROR_WEBCAM,
    max_side=MAX_IMAGE_SIDE,
)
