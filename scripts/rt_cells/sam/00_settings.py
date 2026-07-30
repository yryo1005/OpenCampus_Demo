MIRROR_WEBCAM = True
SAM_MODEL_TYPE = "vit_b"  # vit_b / vit_l / vit_h
MAX_IMAGE_SIDE = 512
# 埋め込み再計算の間隔（フレーム数）．大きいほど速いが追従が遅れる
EMBED_EVERY_N = 4

print(f"SAM_MODEL_TYPE={SAM_MODEL_TYPE}, MAX_IMAGE_SIDE={MAX_IMAGE_SIDE}")
