# prepare_pano.py — положи в корень проекта
from PIL import Image
import sys

src = sys.argv[1] if len(sys.argv) > 1 else r"unn_panorams\328.jpg"
im = Image.open(src).convert("RGB")
print("original size:", im.size)
out = im.resize((1024, 512), Image.LANCZOS)
out.save(r"data\layoutnet_dataset\test\img\328.jpg")
print("saved 1024x512 -> data/layoutnet_dataset/test/img/328.jpg")