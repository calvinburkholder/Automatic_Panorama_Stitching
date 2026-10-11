from __future__ import print_function
from builtins import range
import os
import argparse
 
import numpy as np
from PIL import Image, ImageDraw
from skimage import color
import matplotlib.pyplot as plt

import cv2
 
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def detect_features(image):
    gray_image = np.array(image.convert("L"))
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    return keypoints, descriptors


def match_descriptors(desc1, desc2, r):
    if desc1 is None or desc2 is None:
        return []
    if len(desc2) < 2:
        return []

    matches = []
    for i, descriptor1  in enumerate(desc1):
        differences = desc2 - descriptor1
        distances = np.sqrt(np.sum(differences ** 2, axis=1))

        nearest_indices = np.argsort(distances)[:2]

        first_index = nearest_indices[0]
        second_index = nearest_indices[1]

        d1 = distances[first_index]
        d2 = distances[second_index]

        if d2 > 0 and d1/d2 < r:
            match = cv2.DMatch(
                _queryIdx = i,
                _trainIdx = int(first_index),
                _distance = float(d1)
            )
            matches.append(match)

    return matches

def visualize_matches(img1, kp1, img2, kp2, matches):
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")

    width1, height1 = img1.size
    width2, height2 = img2.size

    # Create a canvas wide enough for both images.
    output = Image.new(
        "RGB",
        (width1 + width2, max(height1, height2)),
        (255, 255, 255)
    )

    # Place both images on the canvas.
    output.paste(img1, (0, 0))
    output.paste(img2, (width1, 0))

    # Create a drawing object.
    draw = ImageDraw.Draw(output)

    for match in matches:
        # Get the keypoint coordinates.
        x1, y1 = kp1[match.queryIdx].pt
        x2, y2 = kp2[match.trainIdx].pt

        # Shift image 2's coordinates to the right side of the canvas.
        x2 += width1

        # Draw a line connecting corresponding keypoints.
        draw.line(
            [(int(x1), int(y1)), (int(x2), int(y2))],
            fill=(0, 255, 0),
            width=1
        )

        # Draw circles at both keypoints.
        radius = 3

        draw.ellipse(
            (int(x1) - radius, int(y1) - radius,
             int(x1) + radius, int(y1) + radius),
            fill=(255, 0, 0)
        )

        draw.ellipse(
            (int(x2) - radius, int(y2) - radius,
             int(x2) + radius, int(y2) + radius),
            fill=(255, 0, 0)
        )

    # Save the visualization using PIL.
    output.save("matches_visualized.png")


img1 = Image.open("IMG_0967.jpg")
img2 = Image.open("IMG_0968.jpg")

kp1, desc1 = detect_features(img1)
kp2, desc2 = detect_features(img2)

matches = match_descriptors(desc1, desc2, .75)
print("Keypoints in image 1:", len(kp1))
print("Keypoints in image 2:", len(kp2))
print("Accepted matches:", len(matches))

visualize_matches(img1, kp1, img2, kp2, matches)

