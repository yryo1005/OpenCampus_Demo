def process_frame(rgb: np.ndarray) -> np.ndarray:
    """顔を検出しスタイル変換した比較画像を返す．

    Args:
        rgb (np.ndarray): (H, W, 3)

    Returns:
        np.ndarray: 変換前|変換後
    """
    box = detect_largest_face(rgb, face_cascade, margin=FACE_MARGIN)
    face = crop_square_face(rgb, box)
    before = np.asarray(
        Image.fromarray(face).resize((PAINT_SIZE, PAINT_SIZE), Image.LANCZOS)
    )
    after = stylize_face(face, style_model, device, PAINT_SIZE)
    return make_side_by_side_rt(before, after)


run_realtime_loop(
    process_frame,
    mirror=MIRROR_WEBCAM,
    max_side=MAX_IMAGE_SIDE,
)
