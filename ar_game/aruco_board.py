"""
aruco_board.py - Helper for detecting an ArUco board and extracting its ROI.

A "board" here is simply four ArUco markers, one in each corner of a rectangular
region. This module finds those four markers, derives the inner quadrilateral
spanned by them and perspectively warps it to a rectangle. The same logic is used
by both AR_game.py and sensor_fusion.py.
"""

import cv2
import cv2.aruco as aruco
import numpy as np

ARUCO_DICT = aruco.DICT_6X6_250
CORNER_MARKER_IDS = [0, 1, 2, 3]


def create_detector():
    """Create and return an ArUco detector for the configured dictionary."""
    dictionary = aruco.getPredefinedDictionary(ARUCO_DICT)
    parameters = aruco.DetectorParameters()
    return aruco.ArucoDetector(dictionary, parameters)


def detect_markers(detector, frame):
    """Detect all markers in ``frame``.

    Returns a dict mapping marker id -> 4x2 corner array, plus the raw
    (corners, ids) for any further use.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    markers = {}
    if ids is not None:
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            markers[int(marker_id)] = marker_corners.reshape(4, 2)
    return markers, corners, ids


def marker_center(marker_corners):
    """Return the center point (x, y) of a single marker's four corners."""
    return marker_corners.mean(axis=0)


def order_points(points):
    """Order four points as top-left, top-right, bottom-right, bottom-left.

    Ordering by actual image position (not by marker id) makes the warp robust
    against however the physical markers happen to be arranged.
    """
    points = np.array(points, dtype="float32")
    ordered = np.zeros((4, 2), dtype="float32")

    # Top-left has the smallest x+y sum, bottom-right the largest.
    s = points.sum(axis=1)
    ordered[0] = points[np.argmin(s)]
    ordered[2] = points[np.argmax(s)]

    # Top-right has the smallest x-y difference, bottom-left the largest.
    diff = np.diff(points, axis=1)
    ordered[1] = points[np.argmin(diff)]
    ordered[3] = points[np.argmax(diff)]

    return ordered


def get_board_quad(markers):
    """Return the inner quadrilateral spanned by the four corner markers.

    We use the inner corner of each marker (the corner pointing towards the
    board center) so that the extracted region lies *between* the markers, as
    required by the assignment. The resulting points are ordered by their image
    position, so the physical arrangement of marker ids does not matter. Returns
    ``None`` if not all four corner markers are visible.
    """
    if not all(mid in markers for mid in CORNER_MARKER_IDS):
        return None

    centers = {mid: marker_center(markers[mid]) for mid in CORNER_MARKER_IDS}
    board_center = np.mean(list(centers.values()), axis=0)

    inner_corners = []
    for mid in CORNER_MARKER_IDS:
        marker_corners = markers[mid]
        # Pick the corner of this marker that is closest to the board center.
        distances = np.linalg.norm(marker_corners - board_center, axis=1)
        inner_corners.append(marker_corners[np.argmin(distances)])

    return order_points(inner_corners)


def warp_board(frame, quad, width, height):
    """Warp the board ``quad`` to a ``width`` x ``height`` rectangle and return
    both the warped image and the perspective matrix used."""
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(frame, matrix, (width, height))
    return warped, matrix
