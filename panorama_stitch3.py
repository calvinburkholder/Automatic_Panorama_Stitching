from __future__ import print_function
import argparse

import numpy as np
from PIL import Image, ImageDraw
import cv2


# ---------------------------------------------------------------------------
# Part 1: Feature detection and matching
# ---------------------------------------------------------------------------

def detect_features(image):
    gray_image = np.array(image.convert("L"))
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    return keypoints, descriptors


def pairwise_distances(a, b):
    """Euclidean distance between every row of a and every row of b."""
    a2 = np.sum(a ** 2, axis=1)[:, None]
    b2 = np.sum(b ** 2, axis=1)[None, :]
    d2 = np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)
    return np.sqrt(d2)


def match_descriptors(desc1, desc2, r):
    """
    Ratio test (d1/d2 < r) in the forward direction, plus a
    mutual-nearest-neighbour check in the reverse direction.
    """
    if desc1 is None or desc2 is None:
        return []
    if len(desc1) < 2 or len(desc2) < 2:
        return []

    desc1 = desc1.astype(np.float64)
    desc2 = desc2.astype(np.float64)

    D = pairwise_distances(desc1, desc2)  # (n1, n2)

    nearest_two = np.argsort(D, axis=1)[:, :2]
    rows = np.arange(len(desc1))
    d1 = D[rows, nearest_two[:, 0]]
    d2 = D[rows, nearest_two[:, 1]]

    reverse_best = np.argmin(D, axis=0)  # best image-1 index for each image-2 descriptor

    matches = []
    for i in range(len(desc1)):
        if d2[i] > 0 and d1[i] / d2[i] < r:
            j = int(nearest_two[i, 0])
            if reverse_best[j] == i:
                matches.append(
                    cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=float(d1[i]))
                )
    return matches


def visualize_matches(img1, kp1, img2, kp2, matches,
                      outlier_matches=None, filename="matches_visualized.png"):
    """Inliers (matches) are drawn green, outliers red."""
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")

    width1, height1 = img1.size
    width2, height2 = img2.size

    output = Image.new(
        "RGB", (width1 + width2, max(height1, height2)), (255, 255, 255)
    )
    output.paste(img1, (0, 0))
    output.paste(img2, (width1, 0))
    draw = ImageDraw.Draw(output)

    def draw_set(match_list, line_color, dot_color):
        radius = 3
        for match in match_list:
            x1, y1 = kp1[match.queryIdx].pt
            x2, y2 = kp2[match.trainIdx].pt
            x2 += width1
            draw.line([(int(x1), int(y1)), (int(x2), int(y2))],
                      fill=line_color, width=1)
            for (cx, cy) in ((x1, y1), (x2, y2)):
                draw.ellipse(
                    (int(cx) - radius, int(cy) - radius,
                     int(cx) + radius, int(cy) + radius),
                    fill=dot_color
                )

    if outlier_matches:
        draw_set(outlier_matches, (255, 0, 0), (255, 128, 0))
    draw_set(matches, (0, 255, 0), (0, 0, 255))

    output.save(filename)


# ---------------------------------------------------------------------------
# Part 2: Geometric transformations (both map image 1 -> image 2)
# ---------------------------------------------------------------------------

def estimate_affine(source, target):
    """Least-squares affine transform from source points to target points."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    if len(source) < 3:
        raise ValueError("At least 3 matches are required.")

    A = []
    B = []
    for (x1, y1), (x2, y2) in zip(source, target):
        A.append([x1, y1, 1, 0, 0, 0])
        A.append([0, 0, 0, x1, y1, 1])
        B.append(x2)
        B.append(y2)

    params, _, _, _ = np.linalg.lstsq(
        np.array(A, dtype=float), np.array(B, dtype=float), rcond=None
    )
    a, b, c, d, e, f = params
    return np.array([[a, b, c],
                     [d, e, f],
                     [0, 0, 1]], dtype=float)


def estimate_homography(source, target):
    """Normalized DLT homography estimate using SVD."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    if len(source) < 4:
        raise ValueError("At least 4 point correspondences are required.")

    def normalize_points(points):
        center = np.mean(points, axis=0)
        centered = points - center
        mean_distance = np.mean(np.linalg.norm(centered, axis=1))
        if mean_distance < 1e-10:
            raise ValueError("Degenerate point configuration.")
        scale = np.sqrt(2) / mean_distance
        T = np.array([
            [scale, 0, -scale * center[0]],
            [0, scale, -scale * center[1]],
            [0, 0, 1]
        ])
        points_h = np.column_stack([points, np.ones(len(points))])
        normalized_h = (T @ points_h.T).T
        return normalized_h[:, :2] / normalized_h[:, 2:3], T

    src_norm, T_src = normalize_points(source)
    dst_norm, T_dst = normalize_points(target)

    A = []
    for (x, y), (xp, yp) in zip(src_norm, dst_norm):
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
    A = np.asarray(A)

    _, _, Vt = np.linalg.svd(A, full_matrices=True)
    H_norm = Vt[-1].reshape(3, 3)

    H = np.linalg.inv(T_dst) @ H_norm @ T_src

    if abs(H[2, 2]) > 1e-10:
        H = H / H[2, 2]
    else:
        H = H / np.linalg.norm(H)
    return H


def apply_transform(points, H):
    """Apply a 3x3 matrix to Nx2 points. Returns (Nx2 points, w values)."""
    points = np.asarray(points, dtype=float)
    ph = (H @ np.column_stack([points, np.ones(len(points))]).T).T
    w = ph[:, 2]
    out = np.full((len(points), 2), np.inf)
    ok = np.abs(w) > 1e-10
    out[ok] = ph[ok, :2] / w[ok, None]
    return out, w


# ---------------------------------------------------------------------------
# Part 3: RANSAC
# ---------------------------------------------------------------------------

def polygon_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_convex(poly):
    signs = []
    n = len(poly)
    for i in range(n):
        a, b, c = poly[i], poly[(i + 1) % n], poly[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(cross)
    signs = np.array(signs)
    return np.all(signs > 0) or np.all(signs < 0)


def is_reasonable_transform(H, image_shape, max_w_ratio=5.0,
                            min_area_ratio=0.1, max_area_ratio=10.0):
    """
    Reject wildly implausible transforms: corners that go near/behind the
    plane at infinity, flip the image, or blow it up / collapse it.
    image_shape is (width, height) of the source image.
    """
    if not np.all(np.isfinite(H)):
        return False

    width, height = image_shape
    corners = np.array([[0, 0], [width - 1, 0],
                        [width - 1, height - 1], [0, height - 1]], dtype=float)

    mapped, w = apply_transform(corners, H)

    if not np.all(np.isfinite(mapped)):
        return False
    if np.any(w <= 0):
        return False
    if w.max() / w.min() > max_w_ratio:
        return False
    if not is_convex(mapped):
        return False

    src_area = polygon_area(corners)
    dst_area = polygon_area(mapped)
    if dst_area * src_area <= 0:  # orientation flip (reflection)
        return False
    ratio = abs(dst_area) / abs(src_area)
    return min_area_ratio < ratio < max_area_ratio


def sample_is_degenerate(points, min_area=1.0):
    """True if any three of the sampled points are (nearly) collinear."""
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = points[i], points[j], points[k]
                area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1])
                                 - (b[1] - a[1]) * (c[0] - a[0]))
                if area < min_area:
                    return True
    return False


def compute_inliers(H, src_pts, dst_pts, threshold):
    """Symmetric transfer error: forward in image 2 and backward in image 1."""
    forward, _ = apply_transform(src_pts, H)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.zeros(len(src_pts), dtype=bool), np.full(len(src_pts), np.inf)
    backward, _ = apply_transform(dst_pts, H_inv)

    err_f = np.linalg.norm(forward - dst_pts, axis=1)
    err_b = np.linalg.norm(backward - src_pts, axis=1)
    errors = np.maximum(err_f, err_b)
    errors[~np.isfinite(errors)] = np.inf
    return errors < threshold, errors


def ransac_transform(src_pts, dst_pts, image_shape, model="homography",
                     N=2000, threshold=5.0, seed=None):
    """
    RANSAC for 'homography' (4-point samples) or 'affine' (3-point samples).
    image_shape is (width, height) of image 1, used for sanity checks.
    """
    if model == "homography":
        k, fit = 4, estimate_homography
    elif model == "affine":
        k, fit = 3, estimate_affine
    else:
        raise ValueError("model must be 'homography' or 'affine'")

    if len(src_pts) < k:
        raise ValueError("Not enough matches for model '%s'." % model)

    rng = np.random.default_rng(seed)
    best_H = None
    best_inliers = None
    best_count = 0

    for _ in range(N):
        idx = rng.choice(len(src_pts), k, replace=False)
        sample_src = src_pts[idx]
        sample_dst = dst_pts[idx]

        if sample_is_degenerate(sample_src) or sample_is_degenerate(sample_dst):
            continue

        try:
            H = fit(sample_src, sample_dst)
        except (ValueError, np.linalg.LinAlgError):
            continue

        if not is_reasonable_transform(H, image_shape):
            continue

        inliers, _ = compute_inliers(H, src_pts, dst_pts, threshold)
        count = int(np.sum(inliers))

        if count > best_count:
            best_count = count
            best_H = H
            best_inliers = inliers

    if best_H is None or best_count < k:
        raise ValueError("RANSAC could not find a valid %s." % model)

    # Refit on all inliers (re-estimate a couple of times as the set grows),
    # but only accept the refit if it is still a sane transform.
    for _ in range(3):
        try:
            refit = fit(src_pts[best_inliers], dst_pts[best_inliers])
        except (ValueError, np.linalg.LinAlgError):
            break
        if not is_reasonable_transform(refit, image_shape):
            break
        new_inliers, _ = compute_inliers(refit, src_pts, dst_pts, threshold)
        if np.sum(new_inliers) < best_count:
            break
        best_H, best_inliers = refit, new_inliers
        best_count = int(np.sum(new_inliers))

    return best_H, best_inliers


# ---------------------------------------------------------------------------
# Part 4: Warping and blending
# ---------------------------------------------------------------------------

def warp_image(image, H, output_width, output_height, chunk_rows=64):
    """
    Inverse-warp a PIL image onto the output canvas. H maps source
    coordinates to canvas coordinates. Processed in row chunks with
    float32 so large canvases do not exhaust memory.
    Returns warped RGB float32 array, coverage mask, feathering weights.
    """
    image = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width = image.shape[:2]

    H_inv = np.linalg.inv(H)

    warped = np.zeros((output_height, output_width, 3), dtype=np.float32)
    mask = np.zeros((output_height, output_width), dtype=bool)
    weights = np.zeros((output_height, output_width), dtype=np.float32)

    col = np.arange(output_width, dtype=np.float64)

    for r0 in range(0, output_height, chunk_rows):
        r1 = min(r0 + chunk_rows, output_height)
        rows = np.arange(r0, r1, dtype=np.float64)

        xx, yy = np.meshgrid(col, rows)
        pts = np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])

        sp = H_inv @ pts
        denom = sp[2]
        valid_denom = np.abs(denom) > 1e-10

        x = np.zeros_like(denom)
        y = np.zeros_like(denom)
        x[valid_denom] = sp[0, valid_denom] / denom[valid_denom]
        y[valid_denom] = sp[1, valid_denom] / denom[valid_denom]

        valid = (
            valid_denom
            & (x >= 0) & (x <= width - 1)
            & (y >= 0) & (y <= height - 1)
        )

        x0 = np.floor(x).astype(int).clip(0, width - 1)
        y0 = np.floor(y).astype(int).clip(0, height - 1)
        x1 = np.minimum(x0 + 1, width - 1)
        y1 = np.minimum(y0 + 1, height - 1)

        dx = (x - x0).astype(np.float32)[:, None]
        dy = (y - y0).astype(np.float32)[:, None]

        pixels = (
            image[y0, x0] * (1 - dx) * (1 - dy)
            + image[y0, x1] * dx * (1 - dy)
            + image[y1, x0] * (1 - dx) * dy
            + image[y1, x1] * dx * dy
        )
        pixels[~valid] = 0

        dist = np.minimum.reduce([x, y, (width - 1) - x, (height - 1) - y])
        w = (np.maximum(dist, 0) + 1).astype(np.float32)
        w[~valid] = 0

        n = r1 - r0
        warped[r0:r1] = pixels.reshape(n, output_width, 3)
        mask[r0:r1] = valid.reshape(n, output_width)
        weights[r0:r1] = w.reshape(n, output_width)

    return warped, mask, weights


def construct_panorama(img1, img2, H, max_pixels=12_000_000):
    """Construct a panorama. H maps img1 coordinates to img2 coordinates."""
    width1, height1 = img1.size
    width2, height2 = img2.size

    corners1 = np.array([[0, 0], [width1 - 1, 0],
                         [width1 - 1, height1 - 1], [0, height1 - 1]], dtype=float)
    corners2 = np.array([[0, 0], [width2 - 1, 0],
                         [width2 - 1, height2 - 1], [0, height2 - 1]], dtype=float)

    transformed_corners1, w = apply_transform(corners1, H)
    if np.any(w <= 0) or not np.all(np.isfinite(transformed_corners1)):
        raise ValueError("Homography maps a corner to infinity or behind the camera.")

    all_corners = np.vstack([transformed_corners1, corners2])

    min_x = int(np.floor(np.min(all_corners[:, 0])))
    min_y = int(np.floor(np.min(all_corners[:, 1])))
    max_x = int(np.ceil(np.max(all_corners[:, 0])))
    max_y = int(np.ceil(np.max(all_corners[:, 1])))

    output_width = max_x - min_x + 1
    output_height = max_y - min_y + 1

    print("Transformed image 1 corners:")
    print(transformed_corners1)
    print("Output canvas: %d x %d (%d pixels)"
          % (output_width, output_height, output_width * output_height))

    if output_width <= 0 or output_height <= 0:
        raise ValueError("Invalid panorama dimensions.")
    if output_width * output_height > max_pixels:
        raise ValueError(
            "Panorama canvas is unexpectedly large. "
            "Check the estimated transform and transformed corners."
        )

    T = np.array([[1, 0, -min_x],
                  [0, 1, -min_y],
                  [0, 0, 1]], dtype=float)

    H1 = T @ H
    H2 = T

    warped1, mask1, weight1 = warp_image(img1, H1, output_width, output_height)
    warped2, mask2, weight2 = warp_image(img2, H2, output_width, output_height)

    weight1 = weight1 * mask1
    weight2 = weight2 * mask2

    total_weight = weight1 + weight2
    safe_total = np.where(total_weight == 0, 1, total_weight)

    panorama = (
        warped1 * weight1[:, :, None] + warped2 * weight2[:, :, None]
    ) / safe_total[:, :, None]
    panorama[total_weight == 0] = 0

    panorama = np.clip(panorama * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(panorama)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_model(model, img1, img2, kp1, kp2, matches, args):
    src_pts = np.array([kp1[m.queryIdx].pt for m in matches], dtype=float)
    dst_pts = np.array([kp2[m.trainIdx].pt for m in matches], dtype=float)

    H, inliers = ransac_transform(
        src_pts, dst_pts, img1.size, model=model,
        N=args.iters, threshold=args.threshold, seed=args.seed
    )

    print("\n=== %s ===" % model)
    print("Matrix:")
    print(H)
    print("Matches: %d, inliers: %d" % (len(src_pts), int(np.sum(inliers))))

    inlier_matches = [m for m, ok in zip(matches, inliers) if ok]
    outlier_matches = [m for m, ok in zip(matches, inliers) if not ok]
    visualize_matches(img1, kp1, img2, kp2, inlier_matches,
                      outlier_matches, "matches_%s.png" % model)

    return construct_panorama(img1, img2, H)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", default="Project2_SampleImages/dvd_left.png")
    parser.add_argument("--right", default="Project2_SampleImages/dvd_right.png")
    parser.add_argument("--ratio", type=float, default=0.75,
                        help="ratio-test threshold r")
    parser.add_argument("--iters", type=int, default=2000, help="RANSAC iterations")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="RANSAC inlier threshold (pixels)")
    parser.add_argument("--model", choices=["homography", "affine", "both"],
                        default="both")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    img1 = Image.open(args.left)
    img2 = Image.open(args.right)

    kp1, desc1 = detect_features(img1)
    kp2, desc2 = detect_features(img2)

    matches = match_descriptors(desc1, desc2, args.ratio)
    print("Keypoints in image 1:", len(kp1))
    print("Keypoints in image 2:", len(kp2))
    print("Accepted matches (r=%.2f): %d" % (args.ratio, len(matches)))

    models = ["homography", "affine"] if args.model == "both" else [args.model]

    for model in models:
        try:
            panorama = run_model(model, img1, img2, kp1, kp2, matches, args)
        except ValueError as e:
            print("\n%s failed: %s" % (model, e))
            continue
        out_name = "panorama_%s.png" % model
        panorama.save(out_name)
        print("Saved %s (%d x %d)" % (out_name, panorama.size[0], panorama.size[1]))


if __name__ == "__main__":
    main()