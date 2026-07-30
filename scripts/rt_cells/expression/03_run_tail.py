def process_frame(rgb: np.ndarray) -> np.ndarray:
    """顔検出＋表情認識結果を描画する．

    Args:
        rgb (np.ndarray): (H, W, 3)

    Returns:
        np.ndarray: 可視化 RGB
    """
    vis = rgb.copy()
    box = detect_largest_face(rgb, face_cascade, margin=FACE_MARGIN)
    if box is None:
        cv2.putText(
            vis,
            "No face",
            (16, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 80, 255),
            2,
            cv2.LINE_AA,
        )
        return vis

    x1, y1, x2, y2 = box
    face = rgb[y1:y2, x1:x2]
    preds = emotion_clf(Image.fromarray(face))
    if isinstance(preds, dict):
        preds = [preds]
    top = max(preds, key=lambda p: float(p["score"]))
    label = str(top["label"]).lower()
    score = float(top["score"])
    en = EMOTION_EN.get(label, label)
    ja = EMOTION_JA.get(label, label)

    cv2.rectangle(vis, (x1, y1), (x2, y2), (40, 200, 80), 2)
    caption = f"{en} {score * 100:.0f}%"
    cv2.rectangle(vis, (x1, max(0, y1 - 36)), (x1 + 12 * len(caption) + 20, y1), (0, 0, 0), -1)
    cv2.putText(
        vis,
        caption,
        (x1 + 8, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    print(f"\r表情: {ja} ({score * 100:.1f}%)", end="", flush=True)
    return vis


run_realtime_loop(
    process_frame,
    mirror=MIRROR_WEBCAM,
    max_side=MAX_IMAGE_SIDE,
)
