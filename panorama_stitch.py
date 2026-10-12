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

def estimate_affine(kp1, kp2, matches):
    if len(matches) < 3:
        raise ValueError("At least 3 matches are required.")

    A = []
    B = []

    for match in matches:
        x1, y1 = kp1[match.queryIdx].pt
        x2, y2 = kp2[match.trainIdx].pt

        A.append([x1, y1, 1, 0, 0, 0])
        A.append([0, 0, 0, x1, y1, 1])

        B.append(x2)
        B.append(y2)

    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)

    # Solve for the six affine parameters.
    params, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    a, b, c, d, e, f = params

    H = np.array([
        [a, b, c],
        [d, e, f],
        [0, 0, 1]
    ])

    return H


# def apply_affine(points, H):
#     num_points = len(points)
#     homogeneous_points = np.zeros((3, num_points))

#     for i in range(num_points):
#         homogeneous_points[0, i] = points[i,0]
#         homogeneous_points[1, i] = points[i,1]
#         homogeneous_points[2, i] = 1

#     transformed = H @ homogeneous_points
#     return transformed[:2, :].T

def estimate_homography(source, target):
    M = 2 * len(source)
    N = 9

    A_hom = np.zeros((M, N))

    for j in range(len(source)):
        x, y = source[j]
        xp, yp = target[j]

        A_hom[2 * j, :] = [
            -x, -y, -1,
             0,  0,  0,
             x * xp, y * xp, xp
        ]

        A_hom[2 * j + 1, :] = [
             0,  0,  0,
            -x, -y, -1,
             x * yp, y * yp, yp
        ]

    # Find the unit-length parameter vector minimizing ||A_hom @ p||^2
    _, _, Vt = np.linalg.svd(A_hom)
    p = Vt[-1, :]

    # Construct the homography matrix from all 9 parameters
    H_estimated = p.reshape(3, 3)

    return H_estimated

# def apply_homography(points, H):
#     num_points = len(points)
#     homogeneous_points = np.zeros((3, num_points))

#     for i in range(num_points):
#         homogeneous_points[0, i] = points[i,0]
#         homogeneous_points[1, i] = points[i,1]
#         homogeneous_points[2, i] = 1

#     transformed = H @ homogeneous_points
#     new_points = np.zeros((num_points, 2))
#     for i in range(num_points):
#         new_points[i, 0] = transformed[0,i] / transformed[2,i]
#         new_points[i, 1] = transformed[1,i] / transformed[2,i]

#     return new_points


