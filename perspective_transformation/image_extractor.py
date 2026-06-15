"""
image_extractor.py - Perspective Transformation (Assignment 4, Task 1)

Loads and displays an image. The user selects four points by clicking into the
image. The selected quadrilateral is then perspectively warped to a rectangle of
the requested target resolution.

Controls:
    Left click  - select a corner point (up to four)
    ESC         - discard current selection / result and start over
    S           - (in the result view) save the warped image to the output path
    Q           - quit the program

Usage:
    python image_extractor.py --input sample_image.jpg --output result.jpg \
        --width 800 --height 600
"""

import argparse
import sys

import cv2
import numpy as np

WINDOW_NAME = "Image Extractor"

POINT_COLOR = (0, 0, 255)      # red
POINT_RADIUS = 6
LINE_COLOR = (0, 255, 0)       # green
LINE_THICKNESS = 2


def parse_arguments():
    """Parse command line parameters for input, output and target resolution."""
    parser = argparse.ArgumentParser(
        description="Extract and perspectively warp a region of an image."
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to the input image file.")
    parser.add_argument("--output", "-o", required=True,
                        help="Path where the warped result will be saved.")
    parser.add_argument("--width", "-W", type=int, default=800,
                        help="Width of the warped result in pixels (default: 800).")
    parser.add_argument("--height", "-H", type=int, default=600,
                        help="Height of the warped result in pixels (default: 600).")
    return parser.parse_args()


def order_points(points):
    """Order four points consistently as top-left, top-right, bottom-right,
    bottom-left.

    This guarantees a correct (non-twisted) perspective transform regardless of
    the order in which the user clicked the corners.
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


def warp_perspective(image, points, width, height):
    """Warp the quadrilateral described by ``points`` to a ``width`` x ``height``
    rectangle."""
    src = order_points(points)
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (width, height))


def draw_overlay(image, points):
    """Return a copy of ``image`` with the currently selected points and the
    connecting polygon drawn on top (visual feedback for the user)."""
    overlay = image.copy()

    for i in range(1, len(points)):
        cv2.line(overlay, points[i - 1], points[i], LINE_COLOR, LINE_THICKNESS)
    if len(points) == 4:
        cv2.line(overlay, points[3], points[0], LINE_COLOR, LINE_THICKNESS)

    for point in points:
        cv2.circle(overlay, point, POINT_RADIUS, POINT_COLOR, -1)

    return overlay


class ImageExtractor:
    """Encapsulates the interactive point selection and warping workflow."""

    def __init__(self, image, width, height, output_path):
        self.image = image
        self.width = width
        self.height = height
        self.output_path = output_path

        self.points = []
        self.result = None

    def on_mouse(self, event, x, y, flags, param):
        """Mouse callback: collect up to four points on left clicks."""
        if event == cv2.EVENT_LBUTTONDOWN and self.result is None:
            if len(self.points) < 4:
                self.points.append((x, y))
                # Compute the warp as soon as the fourth point is set.
                if len(self.points) == 4:
                    self.result = warp_perspective(
                        self.image, self.points, self.width, self.height
                    )

    def reset(self):
        """Discard the current selection and result, starting over."""
        self.points = []
        self.result = None

    def save(self):
        """Save the warped result to the configured output path."""
        if self.result is not None:
            cv2.imwrite(self.output_path, self.result)
            print(f"Saved warped image to '{self.output_path}'")

    def run(self):
        """Main interaction loop."""
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        while True:
            if self.result is not None:
                cv2.imshow(WINDOW_NAME, self.result)
            else:
                cv2.imshow(WINDOW_NAME, draw_overlay(self.image, self.points))

            key = cv2.waitKey(20) & 0xFF

            if key == 27:           # ESC
                self.reset()
            elif key == ord("s"):   # S
                self.save()
            elif key == ord("q"):   # Q
                break

        cv2.destroyAllWindows()


def main():
    args = parse_arguments()

    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: could not load image '{args.input}'", file=sys.stderr)
        sys.exit(1)

    extractor = ImageExtractor(image, args.width, args.height, args.output)
    extractor.run()


if __name__ == "__main__":
    main()
