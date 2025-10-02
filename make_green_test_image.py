from PIL import Image
from pathlib import Path
Path("examples").mkdir(exist_ok=True)
Image.new("RGB", (800, 600), (30, 150, 30)).save("examples/leaf.jpg")
print("saved examples/leaf.jpg")
