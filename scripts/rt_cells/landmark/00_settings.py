MIRROR_WEBCAM = True
MAX_NUM_FACES = 2
DRAW_TESSELATION = True
DRAW_CONTOURS = True
DRAW_IRISES = True
DRAW_LANDMARK_DOTS = False
MAX_IMAGE_SIDE = 640

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = "models/face_landmarker.task"

print(f"MIRROR_WEBCAM={MIRROR_WEBCAM}, MAX_NUM_FACES={MAX_NUM_FACES}")
