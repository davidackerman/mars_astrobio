from PIL import Image
import numpy as np

frames = []
for i in range(4):
    img = Image.open(f'outputs/backyard_worlds_movers/subjects/examined/subject_1904879/frame_{i:02d}.jpg')
    arr = np.array(img.convert('L'))
    frames.append(arr)
    print(f"Frame {i}: {arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")
