"""Axis-aligned box helpers. Boxes are (x1, y1, x2, y2) tuples in pixels."""
from __future__ import annotations

import math
from typing import Sequence

Box = tuple[float, float, float, float]


def center(b: Box) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def width(b: Box) -> float:
    return max(0.0, b[2] - b[0])


def height(b: Box) -> float:
    return max(0.0, b[3] - b[1])


def size(b: Box) -> float:
    """Longer side of a box."""
    return max(width(b), height(b))


def area(b: Box) -> float:
    return width(b) * height(b)


def center_dist(a: Box, b: Box) -> float:
    (ax, ay), (bx, by) = center(a), center(b)
    return math.hypot(ax - bx, ay - by)


def intersection(a: Box, b: Box) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    return iw * ih if iw > 0 and ih > 0 else 0.0


def overlaps(a: Box, b: Box) -> bool:
    return intersection(a, b) > 0


def x_overlap_ratio(a: Box, b: Box) -> float:
    """Horizontal overlap divided by the narrower box width (0..1)."""
    ov = min(a[2], b[2]) - max(a[0], b[0])
    denom = min(width(a), width(b))
    return max(0.0, ov) / denom if denom > 0 else 0.0


def edge_dist(a: Box, b: Box) -> float:
    """Shortest edge-to-edge distance (0 when boxes touch or overlap)."""
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return math.hypot(dx, dy)


def pad(b: Box, p: float) -> Box:
    return (b[0] - p, b[1] - p, b[2] + p, b[3] + p)


def contains_point(b: Box, pt: tuple[float, float]) -> bool:
    return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]


def median_box(boxes: Sequence[Box]) -> Box:
    cols = list(zip(*boxes))
    return tuple(sorted(c)[len(c) // 2] for c in cols)  # type: ignore[return-value]


def short_side(b: Box) -> float:
    return min(width(b), height(b))
