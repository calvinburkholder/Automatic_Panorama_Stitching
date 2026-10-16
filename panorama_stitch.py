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
    a2 = np.sum(a ** 2, axis=1)[:, None]
    b2 = np.sum(b ** 2, axis=1)[None, :]
    d2 = np.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)
    return np.sqrt(d2)


def match_descriptors(desc1, desc2, r):
    if desc1 is None or desc2 is None:
        return []
    if len(desc1) < 2 or len(desc2) < 2:
        return []

    dtype = np.float64 if len(desc1) * len(desc2) <= 20_000_000 else np.float32
    desc1 = desc1.astype(dtype)
    desc2 = desc2.astype(dtype)

    D = pairwise_distances(desc1, desc2)  # (n1, n2)

    two = np.argpartition(D, 1, axis=1)[:, :2]
    rows = np.arange(len(desc1))
    da = D[rows, two[:, 0]]
    db = D[rows, two[:, 1]]
    swap = db < da
    first = np.where(swap, two[:, 1], two[:, 0])
    d1 = np.where(swap, db, da)
    d2 = np.where(swap, da, db)

    reverse_best = np.argmin(D, axis=0)

    matches = []
    for i in range(len(desc1)):
        if d2[i] > 0 and d1[i] / d2[i] < r:
            j = int(first[i])
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
    if dst_area * src_area <= 0: 
        return False
    ratio = abs(dst_area) / abs(src_area)
    return min_area_ratio < ratio < max_area_ratio


def sample_is_degenerate(points, min_area=1.0):
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


def corners_of(img):
    w, h = img.size
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=float)


def construct_panorama_multi(images, transforms, max_pixels=25_000_000,
                             raw_limit=300_000_000):
    mapped_corners = []
    for k, (img, H) in enumerate(zip(images, transforms)):
        mapped, w = apply_transform(corners_of(img), H)
        if np.any(w <= 0) or not np.all(np.isfinite(mapped)):
            raise ValueError(
                "Image %d is mapped through infinity / behind the camera. "
                "The composed transform is not usable." % k
            )
        mapped_corners.append(mapped)

    allc = np.vstack(mapped_corners)
    min_x, min_y = np.floor(allc.min(axis=0))
    max_x, max_y = np.ceil(allc.max(axis=0))

    raw_w = max_x - min_x + 1
    raw_h = max_y - min_y + 1
    print("Raw canvas: %d x %d (%.1f M pixels)"
          % (raw_w, raw_h, raw_w * raw_h / 1e6))

    if raw_w * raw_h > raw_limit:
        raise ValueError(
            "Canvas is absurdly large - a pairwise transform is probably "
            "bad. Check the match visualizations."
        )

    scale = 1.0
    if raw_w * raw_h > max_pixels:
        scale = float(np.sqrt(max_pixels / (raw_w * raw_h)))
        print("Downscaling canvas by %.3f to fit memory" % scale)

    out_w = int(np.ceil((max_x - min_x) * scale)) + 1
    out_h = int(np.ceil((max_y - min_y) * scale)) + 1
    print("Output canvas: %d x %d" % (out_w, out_h))

    S = np.array([[scale, 0, -min_x * scale],
                  [0, scale, -min_y * scale],
                  [0, 0, 1]], dtype=float)

    num = np.zeros((out_h, out_w, 3), dtype=np.float32)
    den = np.zeros((out_h, out_w), dtype=np.float32)

    for k, (img, H) in enumerate(zip(images, transforms)):
        Hc = S @ H
        pts, _ = apply_transform(corners_of(img), Hc)

        x0 = max(0, int(np.floor(pts[:, 0].min())) - 1)
        x1 = min(out_w, int(np.ceil(pts[:, 0].max())) + 2)
        y0 = max(0, int(np.floor(pts[:, 1].min())) - 1)
        y1 = min(out_h, int(np.ceil(pts[:, 1].max())) + 2)
        if x1 <= x0 or y1 <= y0:
            continue

        T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=float)
        warped, mask, weight = warp_image(img, T @ Hc, x1 - x0, y1 - y0)

        weight = weight * mask
        num[y0:y1, x0:x1] += warped * weight[:, :, None]
        den[y0:y1, x0:x1] += weight
        print("  blended image %d into region x[%d:%d] y[%d:%d]"
              % (k, x0, x1, y0, y1))

    safe = np.where(den == 0, 1, den)
    num /= safe[:, :, None]
    num[den == 0] = 0

    return Image.fromarray(np.clip(num * 255, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Part 5: Multiple images
# ---------------------------------------------------------------------------

def normalize_matrix(H):
    if abs(H[2, 2]) > 1e-10:
        return H / H[2, 2]
    return H / np.linalg.norm(H)


def transfer_errors(H, src_pts, dst_pts):
    if H is None:
        return np.full(len(src_pts), np.nan)
    pred, _ = apply_transform(src_pts, H)
    err = np.linalg.norm(pred - dst_pts, axis=1)
    err[~np.isfinite(err)] = np.nan
    return err


def err_summary(err):
    if err.size == 0 or np.all(np.isnan(err)):
        return float("nan"), float("nan")
    return float(np.nanmedian(err)), float(np.nanmean(err))


def estimate_pairwise(images, keypoints, descriptors, model, args):
    pair_transforms = []
    for i in range(len(images) - 1):
        matches = match_descriptors(descriptors[i], descriptors[i + 1], args.ratio)
        print("\nPair %d -> %d: %d matches (r=%.2f)"
              % (i, i + 1, len(matches), args.ratio))

        src_pts = np.array([keypoints[i][m.queryIdx].pt for m in matches], dtype=float)
        dst_pts = np.array([keypoints[i + 1][m.trainIdx].pt for m in matches], dtype=float)

        try:
            H, inliers = ransac_transform(
                src_pts, dst_pts, images[i].size, model=model,
                N=args.iters, threshold=args.threshold, seed=args.seed
            )
        except ValueError as e:
            raise ValueError("Pair %d -> %d: %s" % (i, i + 1, e))

        n_in = int(np.sum(inliers))
        print("  tentative matches: %d" % len(matches))
        print("  inliers: %d / %d (%.1f%%)  outliers: %d"
              % (n_in, len(matches), 100.0 * n_in / len(matches), len(matches) - n_in))

        # Reprojection error (forward, in pixels; T maps image i -> image i+1)
        fit = estimate_homography if model == "homography" else estimate_affine
        try:
            H_before = fit(src_pts, dst_pts)  # non-robust fit on ALL tentative matches
        except (ValueError, np.linalg.LinAlgError):
            H_before = None
        err_before = transfer_errors(H_before, src_pts, dst_pts)
        err_after = transfer_errors(H, src_pts, dst_pts)
        b_med, b_mean = err_summary(err_before)
        i_med, i_mean = err_summary(err_after[inliers])
        a_med, a_mean = err_summary(err_after)
        print("  reprojection error (px):")
        print("    before RANSAC (fit on all tentative, measured on all tentative):"
              "  median %.2f  mean %.2f" % (b_med, b_mean))
        print("    after RANSAC  (final model, measured on inliers only):"
              "           median %.2f  mean %.2f" % (i_med, i_mean))
        print("    after RANSAC  (final model, measured on all tentative):"
              "          median %.2f  mean %.2f" % (a_med, a_mean))
        print(np.array2string(H, precision=4, suppress_small=True))

        inl = [m for m, ok in zip(matches, inliers) if ok]
        outl = [m for m, ok in zip(matches, inliers) if not ok]
        visualize_matches(images[i], keypoints[i], images[i + 1], keypoints[i + 1],
                          inl, outl, "matches_%s_%d_%d.png" % (model, i, i + 1))

        pair_transforms.append(H)
    return pair_transforms


def compose_to_reference(pair_transforms, ref):

    m = len(pair_transforms) + 1
    T = [None] * m
    T[ref] = np.eye(3)

    for i in range(ref - 1, -1, -1):
        T[i] = normalize_matrix(T[i + 1] @ pair_transforms[i])

    for i in range(ref + 1, m):
        T[i] = normalize_matrix(T[i - 1] @ np.linalg.inv(pair_transforms[i - 1]))

    return T


def load_images(args):
    import glob
    import os

    paths = list(args.images)
    if args.dir:
        exts = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG")
        found = []
        for e in exts:
            found.extend(glob.glob(os.path.join(args.dir, e)))
        paths.extend(sorted(set(found)))
    if not paths:
        paths = ["Project2_SampleImages/dvd_left.png",
                 "Project2_SampleImages/dvd_right.png"]

    images = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if args.max_dim and max(img.size) > args.max_dim:
            img.thumbnail((args.max_dim, args.max_dim), Image.LANCZOS)
        print("Loaded %s  size=%s" % (p, img.size))
        images.append(img)
    return paths, images


def main():
    parser = argparse.ArgumentParser(
        description="Stitch an ordered sequence of overlapping images."
    )
    parser.add_argument("images", nargs="*",
                        help="images in left-to-right order")
    parser.add_argument("--dir", default=None,
                        help="folder of images; used in alphabetical order")
    parser.add_argument("--ratio", type=float, default=0.75,
                        help="ratio-test threshold r")
    parser.add_argument("--iters", type=int, default=2000, help="RANSAC iterations")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="RANSAC inlier threshold (pixels)")
    parser.add_argument("--model", choices=["homography", "affine", "both"],
                        default="both")
    parser.add_argument("--ref", type=int, default=None,
                        help="reference image index (default: middle image)")
    parser.add_argument("--max_dim", type=int, default=1600,
                        help="downscale inputs so the long side is at most this (0 = off)")
    parser.add_argument("--output", default="panorama",
                        help="output filename prefix")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths, images = load_images(args)
    if len(images) < 2:
        raise SystemExit("Need at least two images.")

    ref = args.ref if args.ref is not None else len(images) // 2
    if not 0 <= ref < len(images):
        raise SystemExit("--ref must be between 0 and %d" % (len(images) - 1))
    print("Reference image: %d (%s)" % (ref, paths[ref]))

    keypoints, descriptors = [], []
    for img in images:
        kp, desc = detect_features(img)
        keypoints.append(kp)
        descriptors.append(desc)
        print("Keypoints:", len(kp))

    models = ["homography", "affine"] if args.model == "both" else [args.model]

    for model in models:
        print("\n==================== %s ====================" % model)
        try:
            pair_transforms = estimate_pairwise(images, keypoints, descriptors,
                                                model, args)
            to_ref = compose_to_reference(pair_transforms, ref)
            panorama = construct_panorama_multi(images, to_ref)
        except ValueError as e:
            print("\n%s failed: %s" % (model, e))
            continue

        out_name = "%s_%s.png" % (args.output, model)
        panorama.save(out_name)
        print("Saved %s (%d x %d)" % (out_name, panorama.size[0], panorama.size[1]))


if __name__ == "__main__":
    main()