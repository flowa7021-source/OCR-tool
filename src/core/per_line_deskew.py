"""Per-text-row skew correction.

Page-level deskew (see :mod:`src.core.deskew_handler`) rotates the
entire image by a single angle. On scans where different parts of the
page have different skew — e.g. a tax-form page with a header above
and a pre-printed table below that warp independently — that's
insufficient: rotating to fix the table leaves the header crooked
and vice versa.

This module detects individual text rows, estimates each row's own
skew, and rotates the rows independently into a clean output image.
It's engine-agnostic (works on any grayscale page) and plugs into the
preprocessor pipeline as an optional step before binarisation.

Algorithm:
    1. Coarse binarisation (Otsu) to separate text pixels from
       background.
    2. Horizontal projection profile (row-sum of text pixels),
       smoothed over a configurable kernel.
    3. Find text rows as contiguous bands where the smoothed profile
       exceeds a data-driven threshold (median + scaled MAD).
    4. For each row, detect local skew via Hough transform on its
       binarised strip. Clamp to ``max_angle_deg``.
    5. Rotate the original-grayscale row strip by ``-angle``. Paste
       back into the output page at the original y-position, centered
       horizontally to account for expanded width.

The output image has the same shape as the input. Regions between
detected rows are copied verbatim (not rotated), so non-text areas
— lines, stamps, graphics — are preserved exactly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Row:
    """A detected text row with its local skew angle."""
    y0: int           # top of the row (inclusive)
    y1: int           # bottom (exclusive)
    angle_deg: float  # detected local skew


@dataclass
class PerLineDeskewStats:
    rows_detected: int = 0
    rows_rotated: int = 0
    mean_abs_angle: float = 0.0
    max_abs_angle: float = 0.0


def _project_profile(binary: np.ndarray, smooth: int = 15) -> np.ndarray:
    """Row-sum of foreground pixels (255 - image, so text→high)."""
    inv = 255 - binary
    profile = inv.sum(axis=1).astype(np.float32)
    if smooth > 1:
        kernel = np.ones(smooth, dtype=np.float32) / smooth
        profile = np.convolve(profile, kernel, mode="same")
    return profile


def _find_rows(
    profile: np.ndarray,
    min_height: int = 10,
    threshold_scale: float = 1.5,
) -> list[tuple[int, int]]:
    """Return list of (y0, y1) row bounds where the profile is above
    (median + scale × MAD). Adjacent active indices are merged.
    """
    med = float(np.median(profile))
    mad = float(np.median(np.abs(profile - med))) or 1.0
    threshold = med + threshold_scale * mad
    active = profile > threshold
    rows: list[tuple[int, int]] = []
    y = 0
    n = len(profile)
    while y < n:
        if active[y]:
            start = y
            while y < n and active[y]:
                y += 1
            end = y
            if end - start >= min_height:
                rows.append((start, end))
        else:
            y += 1
    return rows


def _estimate_row_skew(
    strip: np.ndarray, max_angle_deg: float = 5.0,
) -> float:
    """Hough-line-based skew detection on a single text strip.

    Returns the angle in degrees to ROTATE BY (i.e., positive rotates
    counter-clockwise). Zero if no reliable angle is found.
    """
    # Edge map — Canny works well on binarised or near-binarised text.
    edges = cv2.Canny(strip, 50, 150, apertureSize=3)
    # HoughLinesP returns (x1, y1, x2, y2) segments; we want near-horizontal ones.
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 720,
        threshold=max(10, strip.shape[1] // 20),
        minLineLength=max(20, strip.shape[1] // 8),
        maxLineGap=10,
    )
    if lines is None:
        return 0.0
    angles: list[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        a = np.degrees(np.arctan2(dy, dx))
        # Only keep near-horizontal lines; skew is small by assumption.
        if abs(a) <= max_angle_deg:
            angles.append(a)
    if not angles:
        return 0.0
    # Median is robust to outlier lines (e.g., decorative rules).
    return float(np.median(angles))


def _rotate_strip(
    strip: np.ndarray, angle_deg: float, border_value: int = 255,
) -> np.ndarray:
    """Rotate ``strip`` by ``angle_deg`` around its centre, expanding
    the canvas so nothing is clipped.
    """
    if abs(angle_deg) < 0.05:
        return strip
    h, w = strip.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos_a = abs(rot[0, 0])
    sin_a = abs(rot[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    rot[0, 2] += (new_w - w) / 2.0
    rot[1, 2] += (new_h - h) / 2.0
    return cv2.warpAffine(
        strip, rot, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def deskew_per_line(
    page: np.ndarray,
    min_row_height: int = 10,
    max_angle_deg: float = 3.0,
    profile_smooth: int = 21,
    row_padding: int = 4,
    threshold_scale: float = 1.5,
) -> tuple[np.ndarray, PerLineDeskewStats]:
    """Detect text rows and individually rotate each one.

    Args:
        page: grayscale uint8 array (HxW).
        min_row_height: rows shorter than this are ignored (noise).
        max_angle_deg: clamp each row's rotation to ±this. Larger is
            likely a false detection.
        profile_smooth: moving-average window for the row projection.
            Longer ≈ smoother profile, better for dense text; shorter
            catches single-line rows.
        row_padding: pixels of vertical padding added above/below each
            detected row before local rotation.
        threshold_scale: how far above the profile's median (in MADs)
            a pixel row must be to count as "in a text row".

    Returns:
        (rotated_page, stats). The output page has the same shape as
        the input. Non-row regions are copied unchanged.
    """
    if page.ndim != 2:
        raise ValueError(f"Expected grayscale 2-D array, got shape {page.shape}")
    h, w = page.shape
    # 1. Coarse binary for row detection.
    _, binary = cv2.threshold(page, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    profile = _project_profile(binary, smooth=profile_smooth)
    rows_bounds = _find_rows(
        profile, min_height=min_row_height, threshold_scale=threshold_scale,
    )
    out = page.copy()
    stats = PerLineDeskewStats(rows_detected=len(rows_bounds))
    angles: list[float] = []

    for y0, y1 in rows_bounds:
        # Add vertical padding so ascenders/descenders survive rotation.
        y0p = max(0, y0 - row_padding)
        y1p = min(h, y1 + row_padding)
        strip = page[y0p:y1p]
        strip_bin = binary[y0p:y1p]
        angle = _estimate_row_skew(strip_bin, max_angle_deg=max_angle_deg)
        if abs(angle) < 0.1:
            continue  # nothing to do — row already level
        rotated = _rotate_strip(strip, -angle)
        # Re-centre rotated strip back into the output at the original y0p
        # position. If the rotated strip is taller/wider than the original,
        # we crop evenly to fit.
        rh, rw = rotated.shape
        target_h = y1p - y0p
        target_w = w
        # Vertical crop
        dy = max(0, (rh - target_h) // 2)
        rotated_cropped = rotated[dy:dy + target_h]
        if rotated_cropped.shape[0] < target_h:
            # Pad bottom with background
            pad = np.full(
                (target_h - rotated_cropped.shape[0], rotated_cropped.shape[1]),
                255, dtype=np.uint8,
            )
            rotated_cropped = np.vstack([rotated_cropped, pad])
        # Horizontal crop / pad
        rh2, rw2 = rotated_cropped.shape
        if rw2 >= target_w:
            dx = (rw2 - target_w) // 2
            final = rotated_cropped[:, dx:dx + target_w]
        else:
            padl = (target_w - rw2) // 2
            padr = target_w - rw2 - padl
            final = np.hstack([
                np.full((rh2, padl), 255, dtype=np.uint8),
                rotated_cropped,
                np.full((rh2, padr), 255, dtype=np.uint8),
            ])
        out[y0p:y1p, :] = final
        stats.rows_rotated += 1
        angles.append(abs(angle))

    if angles:
        stats.mean_abs_angle = float(np.mean(angles))
        stats.max_abs_angle = float(np.max(angles))
    return out, stats


__all__ = ["deskew_per_line", "PerLineDeskewStats", "Row"]
