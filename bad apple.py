# based on minimum window size, 64x64 pixels is viable

from itertools import cycle, product
import os
import json
from typing import Optional
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm
import numpy as np
from ffmpeg import FFmpeg

# whole arrays printed, debug
# np.set_printoptions(threshold=np.inf)

inp = "./assets/bad_apple.mp4"
out = "frames"
max_width = 64
threshold = 255 * 0.4


def frame_to_boxes(im: Image, name):
    w, h = im.size
    ratio = w / h

    im = im.convert("L")
    im = im.resize((max_width, int(max_width / ratio)))
    im = im.point(lambda p: 255 if p > threshold else 0)
    im = im.convert("1")

    pixels = im.load()
    visited = np.zeros(im.size, dtype=bool)

    boxes = []
    work = im.copy().convert("RGB")
    draw = ImageDraw.Draw(work)
    fills = cycle(
        [
            "red",
            "green",
            "blue",
            "orange",
            "yellow",
            "purple",
            "pink",
            "cyan",
            "gray",
            "brown",
            "maroon",
            "hotpink",
            "gold",
            "chocolate",
            "green",
        ]
    )

    while False in visited:
        largest: Optional[tuple[int, int, int, int]] = None

        for x, y in product(range(im.width), range(im.height)):
            if visited[x, y] or pixels[x, y] == 0:
                visited[x, y] = True
                continue

            sublargest: Optional[tuple[int, int]] = None
            widest = im.width - x

            if widest == 0:
                continue

            for h in range(im.height - y):
                for w in range(widest + 1):
                    if (
                        (w == widest)
                        or visited[x + w, y + h]
                        or pixels[x + w, y + h] == 0
                    ):
                        break

                widest = min(widest, w)
                if sublargest is None or (sublargest[0] * sublargest[1]) < (
                    (w) * (h + 1)
                ):
                    sublargest = [w, h + 1]

            if largest is None or (largest[2] * largest[3]) < (
                sublargest[0] * sublargest[1]
            ):
                largest = [x, y, *sublargest]

        if largest is None:
            break

        visited[
            largest[0] : largest[0] + largest[2], largest[1] : largest[1] + largest[3]
        ] = True

        boxes.append(largest)

        box = [
            (largest[0], largest[1]),
            (largest[0] + largest[2] - 1, largest[1] + largest[3] - 1),
        ]
        draw.rectangle(box, fill=next(fills))

    # tqdm.write(f"{len(boxes)=}")

    os.makedirs(f"./{out}", exist_ok=True)

    work.save(f"./{out}/{name}.png")

    return boxes


print("Converting video to frames...")

image_counter = 0

cap = cv2.VideoCapture(inp)
prog = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
all_boxes = []

if not os.path.exists("./assets/bad_apple.ogg") and os.path.exists(
    "./assets/bad_apple.mp4"
):
    ffmpeg = (
        FFmpeg()
        .option("y")
        .input("./assets/bad_apple.mp4")
        .output(
            "./assets/bad_apple.ogg",
            {"codec:a": "libvorbis", "q:a": "0"},
            f="ogg",
            vn=None,
        )
    )

    ffmpeg.execute()

try:
    while cap.isOpened():
        ret, cv2_im = cap.read()
        if ret:
            converted = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)

            pil_im = Image.fromarray(converted)
            all_boxes.append(frame_to_boxes(pil_im, f"{image_counter}"))
            image_counter += 1
            prog.update()
        elif not ret:
            break

    cap.release()
finally:
    with open("./assets/boxes.json", "w") as f:
        json.dump(all_boxes, f)

# checks and such
with open("./assets/boxes.json") as f:
    j = json.load(f)
    print(f"Most visible windows: {max(len(b) for b in j)}")
    print(f"Total frames: {len(j)}")
    print(f"Total window changes: {sum(len(b) for b in j)}")
    print(
        f"Base width: {max(max((coords[0]+coords[2] for coords in b), default=0) for b in j)}\nBase height: {max(max((coords[1]+coords[3] for coords in b), default=0) for b in j)}"
    )

print("Serialising box-o'-bytes to boxes.bin")
with open("./assets/boxes.bin", "wb") as f:
    for frame in j:
        for window in frame:
            f.write(bytes(window))
            # null window signifies new frame
        f.write(bytes([0, 0, 0, 0]))

# im = Image.open('bad apple.jpg')
# frame_to_boxes(im, 'test')
