def process_frame(rgb: np.ndarray) -> np.ndarray:
    """顔ランドマークを検出し可視化する．

    Args:
        rgb (np.ndarray): (H, W, 3)

    Returns:
        np.ndarray: 可視化 RGB
    """
    global _frame_ts_ms
    _frame_ts_ms += 33
    rgb = np.ascontiguousarray(rgb)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_landmarker.detect_for_video(mp_image, _frame_ts_ms)
    vis = rgb.copy()
    Conn = vision.FaceLandmarksConnections
    for landmarks in result.face_landmarks:
        if DRAW_TESSELATION:
            draw_connections(vis, landmarks, Conn.FACE_LANDMARKS_TESSELATION, COLOR_MESH, 1)
        if DRAW_CONTOURS:
            draw_connections(vis, landmarks, Conn.FACE_LANDMARKS_CONTOURS, COLOR_CONTOUR, 2)
        if DRAW_IRISES:
            draw_connections(vis, landmarks, Conn.FACE_LANDMARKS_LEFT_IRIS, COLOR_IRIS, 2)
            draw_connections(vis, landmarks, Conn.FACE_LANDMARKS_RIGHT_IRIS, COLOR_IRIS, 2)
        if DRAW_LANDMARK_DOTS:
            h, w = vis.shape[:2]
            for lm in landmarks:
                x, y = landmark_to_pixel(lm, w, h)
                cv2.circle(vis, (x, y), 1, COLOR_DOT, -1, lineType=cv2.LINE_AA)
    n = len(result.face_landmarks)
    cv2.putText(
        vis,
        f"faces: {n}",
        (16, vis.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 200, 80),
        2,
        cv2.LINE_AA,
    )
    return vis


run_realtime_loop(
    process_frame,
    mirror=MIRROR_WEBCAM,
    max_side=MAX_IMAGE_SIDE,
)
