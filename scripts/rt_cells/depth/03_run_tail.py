def process_frame(rgb: np.ndarray) -> np.ndarray:
    """深度推定して原画像|深度マップを返す．

    Args:
        rgb (np.ndarray): (H, W, 3)

    Returns:
        np.ndarray: 横並び比較
    """
    result = depth_pipe(Image.fromarray(rgb))
    depth_img = result["depth"]
    depth_gray = np.asarray(depth_img if not isinstance(depth_img, Image.Image) else depth_img)
    if depth_gray.shape[:2] != rgb.shape[:2]:
        depth_gray = cv2.resize(
            depth_gray, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC
        )
    depth_color = depth_to_colormap(depth_gray, CMAP)
    return make_side_by_side_rt(rgb, depth_color)


run_realtime_loop(
    process_frame,
    mirror=MIRROR_WEBCAM,
    max_side=MAX_IMAGE_SIDE,
)
