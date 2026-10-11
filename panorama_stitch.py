from __future__ import print_function
from builtins import range
import os
import argparse
 
import numpy as np
from PIL import Image
from skimage import color
import matplotlib.pyplot as plt

import cv2
 
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def detect_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
            match = cv2.Dmatch(
                queryIdx = i,
                trainIdx = int(first_index),
                distance = float(d1)
            )
            matches.append(match)

    return matches

def visualize_matches(img1, kp1, img2, kp2, matches):
    result = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )




