"""Colour-centroid detector for the top-down operator camera.

Deterministic, no model. Pixel -> bench metres uses the camera's pinhole geometry, so a camera that has been
nudged (the cam_offset gotcha) reports every tube shifted by the same amount. The agents have to notice.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

def _hsv(im: np.ndarray):
    """im float32 0..255 -> hue (deg), sat (0..1), val (0..1)."""
    r, g, b = im[..., 0] / 255.0, im[..., 1] / 255.0, im[..., 2] / 255.0
    mx, mn = np.max(im, axis=-1) / 255.0, np.min(im, axis=-1) / 255.0
    d = mx - mn + 1e-9
    h = np.zeros_like(mx)
    h = np.where(mx == r, (60 * ((g - b) / d)) % 360, h)
    h = np.where(mx == g, 60 * ((b - r) / d) + 120, h)
    h = np.where(mx == b, 60 * ((r - g) / d) + 240, h)
    s = np.where(mx > 0, d / (mx + 1e-9), 0)
    return h, s, mx


HUES = {"red": lambda h: (h < 18) | (h > 342), "green": lambda h: (h > 85) & (h < 160), "blue": lambda h: (h > 200) & (h < 260)}
# tubes: saturated solid colour. racks: translucent patch over a near-white floor -> same hue, low saturation, bright.
TUBE = lambda s, v: (s > 0.55) & (v > 0.25)
RACK = lambda s, v: (s > 0.12) & (s < 0.5) & (v > 0.7)


def pixel_to_bench(u: float, v: float, w: int, h: int, cam_pos, fovy_deg: float, obj_z: float = 0.01) -> tuple[float, float]:
    """Top-down camera with xyaxes='0 -1 0  1 0 0': image right = bench -y, image up = bench +x."""
    depth = float(cam_pos[2]) - obj_z
    m_per_px = 2.0 * depth * np.tan(np.radians(fovy_deg) / 2.0) / h
    dx_img = (u - w / 2.0) * m_per_px   # right
    dy_img = (h / 2.0 - v) * m_per_px   # up
    x = float(cam_pos[0]) + dy_img
    y = float(cam_pos[1]) - dx_img
    return round(x, 3), round(y, 3)


def detect(image_path: str, cam_pos, fovy_deg: float, min_pixels: int = 12) -> dict:
    im = np.asarray(Image.open(image_path).convert("RGB")).astype(np.float32)
    h, w, _ = im.shape
    hue, sat, val = _hsv(im)
    tubes, racks = [], []
    for color, band in HUES.items():
        for is_rack, pred in ((False, TUBE), (True, RACK)):
            mask = band(hue) & pred(sat, val)
            n = int(mask.sum())
            if n < (600 if is_rack else min_pixels):   # racks are ~5k px; a tube's anti-aliased edge is ~100
                continue
            vs, us = np.nonzero(mask)
            x, y = pixel_to_bench(us.mean(), vs.mean(), w, h, cam_pos, fovy_deg, obj_z=0.001 if is_rack else 0.03)  # tube top disc
            (racks if is_rack else tubes).append({"color": color, "x": x, "y": y, "pixels": n})
    return {"tubes": tubes, "racks": racks}
