from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


# ---------------------------- Configuration ----------------------------

RATIO_THRESHOLD = 0.70
RANSAC_ITERATIONS = 3000
RANSAC_THRESHOLD = 4.0
MAX_CANVAS_PIXELS = 20_000_000


# ---------------------------- Feature matching -------------------------

def detect_features(image):
    """Return SIFT keypoints and descriptors for a PIL image."""
    gray = np.asarray(image.convert("L"))
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is not None:
        descriptors = descriptors.astype(np.float32)
    return keypoints, descriptors


def match_descriptors(desc1, desc2, ratio=RATIO_THRESHOLD):
    """
    Match descriptors using a two-neighbor ratio test and mutual consistency.
    Each descriptor in image 1 is compared with its two nearest descriptors
    in image 2; a match is kept only if the best match is distinctive and
    the reverse nearest-neighbor match agrees.
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []

    forward = {}
    for i, d1 in enumerate(desc1):
        distances = np.linalg.norm(desc2 - d1, axis=1)
        nearest = np.argsort(distances)[:2]
        j0, j1 = int(nearest[0]), int(nearest[1])
        if distances[j1] > 1e-12 and distances[j0] / distances[j1] < ratio:
            forward[i] = j0

    reverse_best = {}
    for j, d2 in enumerate(desc2):
        distances = np.linalg.norm(desc1 - d2, axis=1)
        reverse_best[j] = int(np.argmin(distances))

    matches = []
    for i, j in forward.items():
        if reverse_best[j] == i:
            distance = float(np.linalg.norm(desc1[i] - desc2[j]))
            matches.append(cv2.DMatch(
                _queryIdx=int(i),
                _trainIdx=int(j),
                _distance=distance
            ))

    return matches


def visualize_matches(img1, kp1, img2, kp2, matches, filename):
    """Save a side-by-side match visualization using PIL."""
    left = img1.convert("RGB")
    right = img2.convert("RGB")
    w1, h1 = left.size
    w2, h2 = right.size
    canvas = Image.new("RGB", (w1 + w2, max(h1, h2)), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (w1, 0))
    draw = ImageDraw.Draw(canvas)

    for match in matches:
        x1, y1 = kp1[match.queryIdx].pt
        x2, y2 = kp2[match.trainIdx].pt
        x2 += w1
        draw.line((int(x1), int(y1), int(x2), int(y2)),
                  fill=(0, 255, 0), width=1)
        r = 3
        draw.ellipse((int(x1)-r, int(y1)-r, int(x1)+r, int(y1)+r),
                     fill=(255, 0, 0))
        draw.ellipse((int(x2)-r, int(y2)-r, int(x2)+r, int(y2)+r),
                     fill=(255, 0, 0))

    canvas.save(filename)
    print(f"Saved match visualization to {filename}")


# ------------------------- Homography estimation -----------------------

def normalize_points(points):
    """Normalize 2D points to mean distance sqrt(2) from the origin."""
    points = np.asarray(points, dtype=float)
    center = np.mean(points, axis=0)
    shifted = points - center
    mean_distance = np.mean(np.linalg.norm(shifted, axis=1))
    if not np.isfinite(mean_distance) or mean_distance < 1e-10:
        raise ValueError("Degenerate point configuration.")
    scale = np.sqrt(2.0) / mean_distance
    T = np.array([
        [scale, 0.0, -scale * center[0]],
        [0.0, scale, -scale * center[1]],
        [0.0, 0.0, 1.0]
    ])
    homogeneous = np.column_stack((points, np.ones(len(points))))
    normalized_h = (T @ homogeneous.T).T
    return normalized_h[:, :2] / normalized_h[:, 2:3], T


def estimate_homography(source, target):
    """
    Estimate H such that target ~ H * source using normalized DLT.
    At least four non-degenerate point pairs are required.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source and target must both have shape (N, 2).")
    if len(source) < 4:
        raise ValueError("At least four correspondences are required.")

    src, T_src = normalize_points(source)
    dst, T_dst = normalize_points(target)

    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
        rows.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])

    A = np.asarray(rows, dtype=float)
    # Reject samples that do not constrain a full homography.
    if np.linalg.matrix_rank(A) < 8:
        raise ValueError("Degenerate correspondence sample.")

    _, _, vt = np.linalg.svd(A, full_matrices=True)
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(T_dst) @ Hn @ T_src

    if abs(H[2, 2]) > 1e-10:
        H /= H[2, 2]
    else:
        norm = np.linalg.norm(H)
        if norm < 1e-12:
            raise ValueError("Invalid homography.")
        H /= norm

    if not np.all(np.isfinite(H)):
        raise ValueError("Non-finite homography.")
    return H
def project_points(points, H):
    """Project N points through H; invalid projections become NaN."""
    points = np.asarray(points, dtype=float)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    projected_h = (H @ homogeneous.T).T
    denom = projected_h[:, 2]
    result = np.full((len(points), 2), np.nan, dtype=float)
    valid = np.abs(denom) > 1e-10
    result[valid] = projected_h[valid, :2] / denom[valid, None]
    result[~np.isfinite(result)] = np.nan
    return result


def corner_denominators_are_safe(H, image_size):
    """
    Reject homographies whose projective horizon crosses or approaches the
    source image. For these neighboring photographs, such transformations
    are not plausible and otherwise create enormous output canvases.
    """
    width, height = image_size
    corners = np.array([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1]
    ], dtype=float)
    homogeneous = np.column_stack((corners, np.ones(4)))
    denominators = (H @ homogeneous.T).T[:, 2]

    # The sign must be consistent throughout the rectangle, and the values
    # must not approach zero relative to the normalized H[2,2].
    if not np.all(np.isfinite(denominators)):
        return False
    if np.any(np.abs(denominators) < 0.08):
        return False
    if np.any(denominators > 0) and np.any(denominators < 0):
        return False

    return True


def plausible_projected_corners(H, image_size, target_size):
    """Reject wildly large corner projections before scoring a hypothesis."""
    if not corner_denominators_are_safe(H, image_size):
        return False

    width, height = image_size
    target_width, target_height = target_size
    corners = np.array([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1]
    ], dtype=float)
    projected = project_points(corners, H)
    if not np.all(np.isfinite(projected)):
        return False

    # Allow generous margins for overlap, but reject near-infinite projections.
    x_limit = 4.0 * max(width, target_width)
    y_limit = 4.0 * max(height, target_height)
    return (np.all(np.abs(projected[:, 0]) < x_limit)
            and np.all(np.abs(projected[:, 1]) < y_limit))

def symmetric_transfer_errors(src_pts, dst_pts, H):
    """Compute symmetric forward/backward reprojection error."""
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(len(src_pts), np.inf)

    projected_dst = project_points(src_pts, H)
    projected_src = project_points(dst_pts, H_inv)
    forward = np.linalg.norm(projected_dst - dst_pts, axis=1)
    backward = np.linalg.norm(projected_src - src_pts, axis=1)
    errors = np.sqrt((forward**2 + backward**2) / 2.0)
    errors[~np.isfinite(errors)] = np.inf
    return errors


def sample_is_degenerate(points):
    """Reject four-point samples with nearly collinear triples."""
    points = np.asarray(points, dtype=float)
    if len(points) != 4:
        return True
    for i in range(4):
        tri = np.delete(points, i, axis=0)
        area2 = abs(
            (tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1])
            - (tri[1, 1] - tri[0, 1]) * (tri[2, 0] - tri[0, 0])
        )
        if area2 < 25.0:
            return True
    return False


def ransac_homography(src_pts, dst_pts, image1_size, image2_size,
                      iterations=RANSAC_ITERATIONS,
                      threshold=RANSAC_THRESHOLD, seed=7):
    """Estimate a robust source-to-target homography with RANSAC."""
    src_pts = np.asarray(src_pts, dtype=float)
    dst_pts = np.asarray(dst_pts, dtype=float)
    if len(src_pts) < 4:
        raise ValueError("Need at least four descriptor matches for RANSAC.")

    rng = np.random.default_rng(seed)
    best_H = None
    best_inliers = None
    best_score = -np.inf

    for _ in range(iterations):
        indices = rng.choice(len(src_pts), 4, replace=False)
        sample_src = src_pts[indices]
        sample_dst = dst_pts[indices]

        if sample_is_degenerate(sample_src) or sample_is_degenerate(sample_dst):
            continue

        try:
            H = estimate_homography(sample_src, sample_dst)
        except (ValueError, np.linalg.LinAlgError):
            continue

        if not plausible_projected_corners(H, image1_size, image2_size):
            continue

        errors = symmetric_transfer_errors(src_pts, dst_pts, H)
        inliers = errors < threshold
        count = int(np.sum(inliers))
        if count < 4:
            continue

        # Prefer more inliers, then smaller median error among those inliers.
        median_error = float(np.median(errors[inliers]))
        score = count - 0.05 * median_error
        if score > best_score:
            best_score = score
            best_H = H
            best_inliers = inliers

    if best_H is None or best_inliers is None or np.sum(best_inliers) < 4:
        raise ValueError(
            "RANSAC could not find a plausible homography. "
            "Inspect matches_visualized.png or try a ratio threshold of 0.75."
        )

    # Refit on all inliers, then recompute inliers and refit once more.
    try:
        refined = estimate_homography(src_pts[best_inliers], dst_pts[best_inliers])
        if plausible_projected_corners(refined, image1_size, image2_size):
            refined_errors = symmetric_transfer_errors(src_pts, dst_pts, refined)
            refined_inliers = refined_errors < threshold
            if np.sum(refined_inliers) >= 4:
                refined2 = estimate_homography(
                    src_pts[refined_inliers], dst_pts[refined_inliers]
                )
                if plausible_projected_corners(
                    refined2, image1_size, image2_size
                ):
                    best_H = refined2
                    best_inliers = (
                        symmetric_transfer_errors(src_pts, dst_pts, best_H)
                        < threshold
                    )
                else:
                    best_H = refined
                    best_inliers = refined_inliers
    except (ValueError, np.linalg.LinAlgError):
        pass

    if np.sum(best_inliers) < 4:
        raise ValueError("Refined homography has too few inliers.")
    return best_H, best_inliers

# ---------------------------- Image warping ---------------------------

def warp_image(image, H_source_to_canvas, output_width, output_height):
    """
    Inverse-map a source image into a canvas using bilinear interpolation.
    Returns RGB pixels (float 0..1), coverage mask, and feather weights.
    """
    source = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = source.shape[:2]
    H_inv = np.linalg.inv(H_source_to_canvas)

    yy, xx = np.indices((output_height, output_width), dtype=np.float64)
    output_h = np.stack(
        (xx.ravel(), yy.ravel(), np.ones(output_width * output_height)),
        axis=0
    )
    source_h = H_inv @ output_h
    denom = source_h[2]

    valid_denom = np.abs(denom) > 1e-10
    x = np.full_like(denom, -1.0)
    y = np.full_like(denom, -1.0)
    x[valid_denom] = source_h[0, valid_denom] / denom[valid_denom]
    y[valid_denom] = source_h[1, valid_denom] / denom[valid_denom]

    valid = (
        valid_denom
        & (x >= 0) & (x <= width - 1)
        & (y >= 0) & (y <= height - 1)
        & np.isfinite(x) & np.isfinite(y)
    )

    # Clamp only for safe array indexing; invalid pixels are zeroed afterward.
    x_safe = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y_safe = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    x0 = np.floor(x_safe).astype(np.int64).clip(0, width - 1)
    y0 = np.floor(y_safe).astype(np.int64).clip(0, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    dx = (x_safe - x0)[:, None].astype(np.float32)
    dy = (y_safe - y0)[:, None].astype(np.float32)

    pixels = (
        source[y0, x0] * (1 - dx) * (1 - dy)
        + source[y0, x1] * dx * (1 - dy)
        + source[y1, x0] * (1 - dx) * dy
        + source[y1, x1] * dx * dy
    )
    pixels[~valid] = 0.0

    distance = np.minimum.reduce([
        x_safe, y_safe, (width - 1) - x_safe, (height - 1) - y_safe
    ])
    weights = np.maximum(distance, 0.0) + 1.0
    weights[~valid] = 0.0

    return (
        pixels.reshape(output_height, output_width, 3),
        valid.reshape(output_height, output_width),
        weights.reshape(output_height, output_width).astype(np.float32)
    )

def construct_panorama(img1, img2, H):
    """Construct a panorama; H maps image 1 coordinates into image 2."""
    width1, height1 = img1.size
    width2, height2 = img2.size
    corners1 = np.array([
        [0, 0], [width1 - 1, 0],
        [width1 - 1, height1 - 1], [0, height1 - 1]
    ], dtype=float)
    corners2 = np.array([
        [0, 0], [width2 - 1, 0],
        [width2 - 1, height2 - 1], [0, height2 - 1]
    ], dtype=float)

    if not corner_denominators_are_safe(H, img1.size):
        raise ValueError(
            "The final homography has an unsafe projective horizon at the "
            "image corners. Check the feature matches."
        )

    transformed1 = project_points(corners1, H)
    if not np.all(np.isfinite(transformed1)):
        raise ValueError("Homography produced non-finite image corners.")

    all_corners = np.vstack((transformed1, corners2))
    min_x = int(np.floor(np.min(all_corners[:, 0])))
    min_y = int(np.floor(np.min(all_corners[:, 1])))
    max_x = int(np.ceil(np.max(all_corners[:, 0])))
    max_y = int(np.ceil(np.max(all_corners[:, 1])))
    out_w = max_x - min_x + 1
    out_h = max_y - min_y + 1
    pixels = out_w * out_h

    print("Image 1 size:", img1.size)
    print("Image 2 size:", img2.size)
    print("Transformed image 1 corners:\n", transformed1)
    print("Output canvas:", out_w, "x", out_h)
    print("Total output pixels:", pixels)

    if out_w <= 0 or out_h <= 0 or pixels > MAX_CANVAS_PIXELS:
        raise ValueError(
            f"Panorama canvas is too large ({pixels:,} pixels). "
            "The homography is not safe to render."
        )

    translation = np.array([
        [1.0, 0.0, -min_x],
        [0.0, 1.0, -min_y],
        [0.0, 0.0, 1.0]
    ])
    H1 = translation @ H
    H2 = translation

    warped1, mask1, weights1 = warp_image(img1, H1, out_w, out_h)
    warped2, mask2, weights2 = warp_image(img2, H2, out_w, out_h)

    weights1 *= mask1
    weights2 *= mask2
    total = weights1 + weights2
    safe_total = np.where(total > 0, total, 1.0)

    result = (
        warped1 * weights1[:, :, None]
        + warped2 * weights2[:, :, None]
    ) / safe_total[:, :, None]
    result[total == 0] = 0.0

    return Image.fromarray(
        np.clip(result * 255.0, 0, 255).astype(np.uint8)
    )

# -------------------------------- Main ---------------------------------

def resolve_image_paths():
    """Read optional command-line paths or use the repository defaults."""
    if len(sys.argv) >= 3:
        return Path(sys.argv[1]), Path(sys.argv[2])

    candidates = [
        (Path("Project2_SampleImages/dvd_left.png"),
         Path("Project2_SampleImages/dvd_right.png")),
        (Path("dvd_left.png"), Path("dvd_right.png")),
    ]
    for left, right in candidates:
        if left.exists() and right.exists():
            return left, right

    raise FileNotFoundError(
        "Could not find the sample images. Run from the repository root, "
        "or pass both image paths: python panorama_stitch.py left.png right.png"
    )


def main():
    path1, path2 = resolve_image_paths()
    img1 = Image.open(path1).convert("RGB")
    img2 = Image.open(path2).convert("RGB")
    print("Image 1:", path1)
    print("Image 2:", path2)

    kp1, desc1 = detect_features(img1)
    kp2, desc2 = detect_features(img2)
    print("Keypoints in image 1:", len(kp1))
    print("Keypoints in image 2:", len(kp2))

    matches = match_descriptors(desc1, desc2)
    print("Accepted matches:", len(matches))
    if len(matches) < 4:
        raise ValueError(
            "Too few matches. Try setting RATIO_THRESHOLD to 0.75."
        )

    src_pts = np.asarray(
        [kp1[m.queryIdx].pt for m in matches], dtype=float
    )
    dst_pts = np.asarray(
        [kp2[m.trainIdx].pt for m in matches], dtype=float
    )

    H, inliers = ransac_homography(
        src_pts, dst_pts, img1.size, img2.size
    )
    print("Homography (image 1 -> image 2):\n", H)
    print("Number of matches:", len(matches))
    print("Number of inliers:", int(np.sum(inliers)))
    print("Corner denominators:",
          (H @ np.array([
              [0, img1.height - 1, img1.width - 1, 0],
              [0, 0, img1.height - 1, img1.height - 1],
              [1, 1, 1, 1]
          ], dtype=float))[2])

    inlier_matches = [
        match for match, keep in zip(matches, inliers) if keep
    ]
    visualize_matches(
        img1, kp1, img2, kp2, matches, "matches_all.png"
    )
    visualize_matches(
        img1, kp1, img2, kp2, inlier_matches, "matches_visualized.png"
    )

    panorama = construct_panorama(img1, img2, H)
    panorama.save("panorama.png")
    print("Saved panorama to panorama.png")
    print("Panorama dimensions:", panorama.size)


if __name__ == "__main__":
    main()
