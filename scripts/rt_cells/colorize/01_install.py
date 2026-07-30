from pathlib import Path

DDCOLOR_DIR = Path("DDColor")
if not DDCOLOR_DIR.exists():
    !git clone --depth 1 https://github.com/piddnad/DDColor.git
else:
    print(f"既に存在します: {DDCOLOR_DIR.resolve()}")

!pip install -q -U "tqdm" "huggingface_hub"
print("インストール完了")
