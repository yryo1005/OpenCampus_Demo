# Webカメラの左右反転（自撮り表示）
MIRROR_WEBCAM = True

# リアルタイム向け: 軽量モデル推奨
#   ddcolor_paper_tiny / ddcolor_modelscope / ddcolor_artistic / ddcolor_paper
MODEL_NAME = "ddcolor_paper_tiny"
INPUT_SIZE = 256
MAX_IMAGE_SIDE = 480

print(f"MIRROR_WEBCAM={MIRROR_WEBCAM}, MODEL_NAME={MODEL_NAME}, INPUT_SIZE={INPUT_SIZE}")
