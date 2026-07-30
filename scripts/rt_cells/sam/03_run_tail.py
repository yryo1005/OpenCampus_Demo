def process_frame(rgb: np.ndarray) -> np.ndarray:
    """中央点プロンプトで SAM セグメンテーションする．

    Args:
        rgb (np.ndarray): (H, W, 3)

    Returns:
        np.ndarray: オーバーレイ結果
    """
    global _sam_frame_i, _last_mask
    h, w = rgb.shape[:2]
    cx, cy = w // 2, h // 2

    if _sam_frame_i % max(int(EMBED_EVERY_N), 1) == 0:
        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[cx, cy]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        _last_mask = masks[best]
    _sam_frame_i += 1

    vis = rgb.copy()
    if _last_mask is not None:
        if _last_mask.shape[:2] != (h, w):
            mask = cv2.resize(
                _last_mask.astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            mask = _last_mask
        vis = overlay_mask(vis, mask)
    cv2.drawMarker(
        vis,
        (cx, cy),
        (255, 255, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=2,
    )
    return vis


run_realtime_loop(
    process_frame,
    mirror=MIRROR_WEBCAM,
    max_side=MAX_IMAGE_SIDE,
)
