from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _detector(method: str):
    method = method.upper()
    if method == "SIFT":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("Your OpenCV build does not include SIFT. Install a recent opencv-python package.")
        return cv2.SIFT_create(nfeatures=6000), cv2.NORM_L2
    if method == "ORB":
        return cv2.ORB_create(nfeatures=6000), cv2.NORM_HAMMING
    raise ValueError("method must be 'SIFT' or 'ORB'")


def create_feature_match_visualization(
    image_a_path: Path,
    image_b_path: Path,
    method: str = "SIFT",
    ratio: float = 0.75,
) -> tuple[np.ndarray, dict]:
    """Extract and match features, then draw good matches with RANSAC inliers highlighted."""
    image_a = cv2.imread(str(image_a_path), cv2.IMREAD_COLOR)
    image_b = cv2.imread(str(image_b_path), cv2.IMREAD_COLOR)
    if image_a is None or image_b is None:
        raise ValueError("Could not load one or both images.")

    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    detector, norm = _detector(method)
    keypoints_a, descriptors_a = detector.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = detector.detectAndCompute(gray_b, None)

    if descriptors_a is None or descriptors_b is None or len(keypoints_a) < 2 or len(keypoints_b) < 2:
        raise RuntimeError("Not enough detectable features in these images. Try more textured or sharper images.")

    matcher = cv2.BFMatcher(norm, crossCheck=False)
    pairs = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]

    inlier_matches = good
    inlier_count = 0
    if len(good) >= 8:
        points_a = np.float32([keypoints_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        points_b = np.float32([keypoints_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findFundamentalMat(points_a, points_b, cv2.FM_RANSAC, 2.0, 0.99)
        if mask is not None:
            mask_flat = mask.ravel().astype(bool)
            inlier_matches = [m for m, keep in zip(good, mask_flat) if keep]
            inlier_count = len(inlier_matches)

    preview_matches = inlier_matches if inlier_matches else good
    drawn = cv2.drawMatches(
        image_a,
        keypoints_a,
        image_b,
        keypoints_b,
        preview_matches[:250],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    stats = {
        "algorithm": method.upper(),
        "keypoints_image_1": len(keypoints_a),
        "keypoints_image_2": len(keypoints_b),
        "ratio_test_matches": len(good),
        "ransac_inliers": inlier_count if inlier_count else None,
        "matches_drawn": min(len(preview_matches), 250),
        "note": "The visualization is an educational pairwise comparison; COLMAP runs its own matching pipeline.",
    }
    return drawn, stats
