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

    if len(desc1) < 2 or len(desc2) < 2:
        return []

    # Find the two nearest neighbors in image 2 for each descriptor in image 1.
    forward_matches = {}

    for i, descriptor1 in enumerate(desc1):
        distances = np.linalg.norm(desc2 - descriptor1, axis=1)
        nearest = np.argsort(distances)[:2]

        first_idx, second_idx = nearest
        d1 = distances[first_idx]
        d2 = distances[second_idx]

        if d2 > 0 and d1 / d2 < r:
            forward_matches[i] = int(first_idx)

    # Find each descriptor in image 2's nearest descriptor in image 1.
    reverse_best = {}

    for j, descriptor2 in enumerate(desc2):
        distances = np.linalg.norm(desc1 - descriptor2, axis=1)
        reverse_best[j] = int(np.argmin(distances))

    # Keep only matches that agree in both directions.
    matches = []

    for i, j in forward_matches.items():
        if reverse_best[j] == i:
            distance = float(np.linalg.norm(desc1[i] - desc2[j]))
            matches.append(
                cv2.DMatch(
                    _queryIdx=i,
                    _trainIdx=j,
                    _distance=distance
                )
            )

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


img1 = Image.open("Project2_SampleImages/dvd_left.png")
img2 = Image.open("Project2_SampleImages/dvd_right.png")

kp1, desc1 = detect_features(img1)
kp2, desc2 = detect_features(img2)

matches = match_descriptors(desc1, desc2, .6)
print("Keypoints in image 1:", len(kp1))
print("Keypoints in image 2:", len(kp2))
print("Accepted matches:", len(matches))

# visualize_matches(img1, kp1, img2, kp2, matches)

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
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    if len(source) < 4:
        raise ValueError("At least 4 point correspondences are required.")

    def normalize_points(points):
        center = np.mean(points, axis=0)
        centered = points - center

        mean_distance = np.mean(
            np.linalg.norm(centered, axis=1)
        )

        if mean_distance < 1e-10:
            raise ValueError("Degenerate point configuration.")

        scale = np.sqrt(2) / mean_distance

        T = np.array([
            [scale, 0, -scale * center[0]],
            [0, scale, -scale * center[1]],
            [0, 0, 1]
        ])

        points_h = np.column_stack([
            points, np.ones(len(points))
        ])

        normalized_h = (T @ points_h.T).T
        normalized = (
            normalized_h[:, :2] / normalized_h[:, 2:3]
        )

        return normalized, T

    src_norm, T_src = normalize_points(source)
    dst_norm, T_dst = normalize_points(target)

    A = []

    for (x, y), (xp, yp) in zip(src_norm, dst_norm):
        A.append([
            -x, -y, -1,
            0, 0, 0,
            x * xp, y * xp, xp
        ])

        A.append([
            0, 0, 0,
            -x, -y, -1,
            x * yp, y * yp, yp
        ])

    A = np.asarray(A)

    _, _, Vt = np.linalg.svd(A, full_matrices=True)
    H_norm = Vt[-1].reshape(3, 3)

    # Convert from normalized coordinates back to pixel coordinates.
    H = np.linalg.inv(T_dst) @ H_norm @ T_src

    # Normalize the matrix for easier inspection.
    if abs(H[2, 2]) > 1e-10:
        H = H / H[2, 2]
    else:
        H = H / np.linalg.norm(H)

    return H

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

def ransac_homography(src_pts, dst_pts, N=1000, threshold=5.0):
    """
    Estimate a homography using RANSAC.
    """
    best_H = None
    best_inliers = None
    best_count = 0

    # Step 1: Randomly choose 4 point correspondences
    for _ in range(N):
        indices = np.random.choice(len(src_pts), 4, replace=False)

        sample_src = src_pts[indices]
        sample_dst = dst_pts[indices]

        # Step 2: Fit a homography to the sample
        H = estimate_homography(sample_src, sample_dst)

        h, w = 1084, 1434  # Image 1 dimensions
        corners = np.array([
            [0, 0, 1],
            [w - 1, 0, 1],
            [w - 1, h - 1, 1],
            [0, h - 1, 1]
        ], dtype=float)

        projected_h = (H @ corners.T).T
        denominators = projected_h[:, 2]

        # Since H[2, 2] is normalized to 1, all four
        # denominators should stay safely positive.
        if np.any(denominators < 0.1):
            continue

        # Map all source points using the hypothesized homography
        src_hom = np.column_stack(
            (src_pts, np.ones(len(src_pts)))
        )
        projected_hom = (H @ src_hom.T).T

        # Convert homogeneous coordinates to (x, y)
        denom = projected_hom[:, 2]
        valid = np.abs(denom) > 1e-10

        projected = np.full_like(dst_pts, np.inf, dtype=float)
        projected[valid] = (
            projected_hom[valid, :2] / denom[valid, None]
        )

        # Step 3: Count the inliers
        errors = np.linalg.norm(projected - dst_pts, axis=1)
        inliers = errors < threshold
        count = np.sum(inliers)

        # Step 5: Keep the hypothesis with the most inliers
        if count > best_count:
            best_count = count
            best_H = H
            best_inliers = inliers

    if best_H is None or best_count < 4:
        raise ValueError("Could not find a valid homography.")

    # Final step from lecture: refit using all inliers
    best_H = estimate_homography(
        src_pts[best_inliers],
        dst_pts[best_inliers]
    )

    return best_H, best_inliers

src_pts = np.array(
    [kp1[match.queryIdx].pt for match in matches],
    dtype=float
)

dst_pts = np.array(
    [kp2[match.trainIdx].pt for match in matches],
    dtype=float
)

H, inliers = ransac_homography(
    src_pts, dst_pts,
    N=1000,
    threshold=5.0
)

print("Homography:")
print(H)
print("Number of matches:", len(src_pts))
print("Number of inliers:", np.sum(inliers))

width, height = img1.size

corners = np.array([
    [0, 0, 1],
    [width - 1, 0, 1],
    [width - 1, height - 1, 1],
    [0, height - 1, 1]
], dtype=float)

print("Homography denominators at image corners:")
print((H @ corners.T).T[:, 2])

inlier_matches = [
    match
    for match, is_inlier in zip(matches, inliers)
    if is_inlier
]

print("Visualizing", len(inlier_matches), "inlier matches")

visualize_matches(
    img1, kp1,
    img2, kp2,
    inlier_matches
)


def warp_image(image, H, output_width, output_height):
    """
    Warp a PIL image into a common output canvas using H,
    which maps source-image coordinates to canvas coordinates.

    Returns:
        warped: warped RGB image as a float array
        mask: pixels covered by the source image
        weights: feathering weights for blending
    """
    image = np.asarray(image.convert("RGB"), dtype=float) / 255.0
    height, width = image.shape[:2]

    # Inverse mapping: for each output pixel, find its source location.
    H_inv = np.linalg.inv(H)

    xx, yy = np.meshgrid(
        np.arange(output_width),
        np.arange(output_height)
    )

    output_points = np.stack([
        xx.ravel(),
        yy.ravel(),
        np.ones(output_width * output_height)
    ])

    source_points = H_inv @ output_points

    denom = source_points[2]
    valid_denom = np.abs(denom) > 1e-10

    x = np.zeros_like(denom)
    y = np.zeros_like(denom)

    x[valid_denom] = source_points[0, valid_denom] / denom[valid_denom]
    y[valid_denom] = source_points[1, valid_denom] / denom[valid_denom]

    # Keep coordinates inside the source image.
    valid = (
        valid_denom
        & (x >= 0) & (x <= width - 1)
        & (y >= 0) & (y <= height - 1)
    )

    # Bilinear interpolation.
    x0 = np.floor(x).astype(int).clip(0, width - 1)
    y0 = np.floor(y).astype(int).clip(0, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)

    dx = (x - x0)[:, None]
    dy = (y - y0)[:, None]

    pixels = (
        image[y0, x0] * (1 - dx) * (1 - dy)
        + image[y0, x1] * dx * (1 - dy)
        + image[y1, x0] * (1 - dx) * dy
        + image[y1, x1] * dx * dy
    )

    pixels[~valid] = 0

    warped = pixels.reshape(output_height, output_width, 3)
    mask = valid.reshape(output_height, output_width)

    # Feathering: give more weight to pixels farther from image edges.
    distance_to_edge = np.minimum.reduce([
        x,
        y,
        (width - 1) - x,
        (height - 1) - y
    ])

    weights = np.maximum(distance_to_edge, 0) + 1
    weights[~valid] = 0
    weights = weights.reshape(output_height, output_width)

    return warped, mask, weights


def construct_panorama(img1, img2, H):
    """
    Construct a panorama. H maps img1 coordinates to img2 coordinates.
    """
    width1, height1 = img1.size
    width2, height2 = img2.size

    # Four corners of each image, in (x, y) coordinates.
    corners1 = np.array([
        [0, 0], [width1 - 1, 0],
        [width1 - 1, height1 - 1], [0, height1 - 1]
    ], dtype=float)

    corners2 = np.array([
        [0, 0], [width2 - 1, 0],
        [width2 - 1, height2 - 1], [0, height2 - 1]
    ], dtype=float)

    # Transform image 1's corners into image 2's coordinate system.
    corners1_h = np.column_stack([
        corners1, np.ones(len(corners1))
    ])
    transformed_h = (H @ corners1_h.T).T

    if np.any(np.abs(transformed_h[:, 2]) < 1e-10):
        raise ValueError("Homography maps a corner to infinity.")

    transformed_corners1 = (
        transformed_h[:, :2] / transformed_h[:, 2:3]
    )

    # Find bounds containing both images.
    all_corners = np.vstack([transformed_corners1, corners2])

    min_x = int(np.floor(np.min(all_corners[:, 0])))
    min_y = int(np.floor(np.min(all_corners[:, 1])))
    max_x = int(np.ceil(np.max(all_corners[:, 0])))
    max_y = int(np.ceil(np.max(all_corners[:, 1])))

    output_width = max_x - min_x + 1
    output_height = max_y - min_y + 1

    print("Image 1 size:", img1.size)
    print("Image 2 size:", img2.size)
    print("Transformed image 1 corners:")
    print(transformed_corners1)
    print("Output canvas width:", output_width)
    print("Output canvas height:", output_height)
    print("Total output pixels:", output_width * output_height)

    if output_width <= 0 or output_height <= 0:
        raise ValueError("Invalid panorama dimensions.")

    if output_width * output_height > 20_000_000:
        raise ValueError(
        "Panorama canvas is unexpectedly large. "
        "Check the estimated homography and transformed corners."
        )

    # Translate the canvas so all coordinates are nonnegative.
    T = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ], dtype=float)

    H1 = T @ H
    H2 = T  # Image 2 already defines the reference coordinate system.

    # Warp both images onto the expanded canvas.
    warped1, mask1, weight1 = warp_image(
        img1, H1, output_width, output_height
    )
    warped2, mask2, weight2 = warp_image(
        img2, H2, output_width, output_height
    )

    # Weighted feathering blend.
    weight1 = weight1 * mask1
    weight2 = weight2 * mask2

    total_weight = weight1 + weight2
    safe_total = np.where(total_weight == 0, 1, total_weight)

    panorama = (
        warped1 * weight1[:, :, None]
        + warped2 * weight2[:, :, None]
    ) / safe_total[:, :, None]

    # Keep uncovered pixels black.
    panorama[total_weight == 0] = 0

    panorama = np.clip(panorama * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(panorama)

panorama = construct_panorama(img1, img2, H)
panorama.save("panorama.png")

print("Saved panorama to panorama.png")
print("Panorama dimensions:", panorama.size)