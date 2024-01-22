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


def frame_to_boxes(im: Image, name: str) -> list[tuple[int, int, int, int]]:
    # Get the width and height of the image
    w, h = im.size

    # Calculate the aspect ratio of the image
    ratio = w / h

    # Convert the image to grayscale
    im = im.convert("L")

    # Resize the image while maintaining the aspect ratio
    max_width = 66  # Maximum width allowed
    im = im.resize((max_width, int(max_width / ratio)))

    # Convert the image to black and white using a threshold value
    threshold = 128  # Threshold value for black and white conversion
    im = im.point(lambda p: 255 if p > threshold else 0)
    im = im.convert("1")

    # Load the pixel data of the image
    pixels = im.load()

    # Create a boolean array to keep track of visited pixels
    visited = np.zeros(im.size, dtype=bool)

    # Create an empty list to store the bounding boxes
    boxes = []

    # Create a copy of the image for drawing the bounding boxes
    work = Image.new("RGB", im.size)
    draw = ImageDraw.Draw(work)

    # Create an iterator for generating fill colors for the bounding boxes
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

    # Iterate until all pixels have been visited
    while False in visited:
        # Initialize the largest bounding box
        largest: Optional[tuple[int, int, int, int]] = None

        # Iterate over each pixel in the image
        for x, y in product(range(im.width), range(im.height)):
            # Skip visited pixels and black pixels
            if visited[x, y] or pixels[x, y] == 0:
                visited[x, y] = True
                continue

            # Initialize the largest sub-bounding box
            sublargest: Optional[tuple[int, int]] = None

            # Initialize the widest width of the sub-bounding box
            widest = im.width - x

            # Iterate over possible heights and widths of the sub-bounding box
            for h in range(im.height - y):
                for w in range(widest + 1):
                    # Break the loop if the pixel is visited or black
                    if (
                        (w == widest)
                        or visited[x + w, y + h]
                        or pixels[x + w, y + h] == 0
                    ):
                        break

                # Update the widest width of the sub-bounding box
                widest = min(widest, w)

                # Update the largest sub-bounding box if necessary
                if sublargest is None or (sublargest[0] * sublargest[1]) < (
                    (w) * (h + 1)
                ):
                    sublargest = [w, h + 1]

            # Update the largest bounding box if necessary
            if largest is None or (largest[2] * largest[3]) < (
                sublargest[0] * sublargest[1]
            ):
                largest = [x, y, *sublargest]

        # Break the loop if no more bounding boxes can be found
        if largest is None:
            break

        # Mark the pixels of the largest bounding box as visited
        visited[
            largest[0] : largest[0] + largest[2], largest[1] : largest[1] + largest[3]
        ] = True

        # Add the largest bounding box to the list of boxes
        boxes.append(largest)

        # Draw the largest bounding box on the copy of the image
        box = [
            (largest[0], largest[1]),
            (largest[0] + largest[2] - 1, largest[1] + largest[3] - 1),
        ]
        draw.rectangle(box, fill=next(fills))

    # Create the output directory if it doesn't exist
    os.makedirs(f"./{out}", exist_ok=True)

    # Save the copy of the image with bounding boxes
    work.save(f"./{out}/{name}.png")

    # Return the list of bounding boxes
    return boxes


print("Converting video to frames...")

if not os.path.exists("./assets/bad_apple.ogg") and os.path.exists(inp):
    ffmpeg = (
        FFmpeg()
        .option("y")
        .input(inp)
        .output(
            "./assets/bad_apple.ogg",
            {"codec:a": "libvorbis", "q:a": "0"},
            f="ogg",
            vn=None,
        )
    )

    ffmpeg.execute()


def process_frame(frame, image_counter):
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(converted)
    boxes = frame_to_boxes(pil_im, f"{image_counter}")
    return boxes


def save_boxes(boxes):
    with open("./assets/boxes.json", "w") as f:
        json.dump(boxes, f)


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    prog = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    all_boxes = []
    image_counter = 0

    try:
        while cap.isOpened():
            ret, cv2_im = cap.read()
            if ret:
                all_boxes.append(process_frame(cv2_im, image_counter))
                image_counter += 1
                prog.update()
            else:
                break
    finally:
        cap.release()
        save_boxes(all_boxes)


# Call the function with the path to your video file
process_video(inp)

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
