"""
AR_game.py - AR Game (Assignment 4, Task 2)

Reads the webcam image, detects a board with one ArUco marker in each corner,
extracts and perspectively warps the region between the markers to the webcam
resolution and displays it in a pyglet application.

Game mechanics
--------------
Targets (circles) spawn at random positions inside the board. The player tracks
an object with their *finger* (detected via motion + skin color, NOT through an
ArUco marker). Touching a target with the finger destroys it and increases the
score. A new target spawns after each hit. The goal is to pop as many targets as
possible.

Controls:
    SPACE - reset the score
    ESC   - quit

Usage:
    python AR_game.py [video_id]
"""

import random
import sys

import cv2
import numpy as np
import pyglet

import aruco_board

# How long (in frames) a target stays before it is replaced.
TARGET_LIFETIME = 150
TARGET_RADIUS = 30


def cv2glet(img):
    """Convert an OpenCV BGR image to a pyglet image."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rows, cols, channels = img_rgb.shape
    raw_img = img_rgb.tobytes()
    return pyglet.image.ImageData(
        width=cols, height=rows, fmt='RGB',
        data=raw_img, pitch=-channels * cols,
    )


class FingerTracker:
    """Tracks a moving, skin-colored object (a finger) inside a frame.

    Combines a background subtractor (to find what is moving) with a simple skin
    color filter (to reject moving shadows / non-hand motion). Returns the
    centroid of the largest matching blob.
    """

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False
        )

    def track(self, frame):
        """Return (x, y) of the tracked finger tip, or None if nothing found."""
        motion = self.bg_subtractor.apply(frame)

        # Skin color mask in YCrCb space (robust to lighting).
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

        # A finger is both moving and skin-colored.
        mask = cv2.bitwise_and(motion, skin)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 400:
            return None

        # Topmost contour point as the finger tip (more intuitive than centroid).
        tip = tuple(largest[largest[:, :, 1].argmin()][0])
        return int(tip[0]), int(tip[1])


class Target:
    """A circular target the player must hit."""

    def __init__(self, width, height):
        self.radius = TARGET_RADIUS
        self.respawn(width, height)

    def respawn(self, width, height):
        margin = self.radius
        self.x = random.randint(margin, width - margin)
        self.y = random.randint(margin, height - margin)
        self.age = 0

    def is_hit(self, point):
        """Return True if ``point`` (x, y) is inside the target."""
        if point is None:
            return False
        dx = point[0] - self.x
        dy = point[1] - self.y
        return dx * dx + dy * dy <= self.radius * self.radius


class ARGame:
    """The AR game application."""

    def __init__(self, video_id=0):
        self.cap = cv2.VideoCapture(video_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam {video_id}")

        # Request a moderate capture resolution for good performance; the webcam
        # will pick the closest mode it supports.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        self.detector = aruco_board.create_detector()
        self.tracker = FingerTracker()
        self.target = Target(self.width, self.height)
        self.score = 0
        self.finger = None

        # The board is warped at full webcam resolution, but the window is scaled
        # down so it never exceeds the screen on high-res webcams.
        win_w, win_h = self._display_size(self.width, self.height)
        self.window = pyglet.window.Window(win_w, win_h, caption="AR Game")
        self.score_label = pyglet.text.Label(
            "", font_size=18, x=10, y=win_h - 10,
            anchor_x="left", anchor_y="top", color=(255, 255, 0, 255),
        )
        self.info_label = pyglet.text.Label(
            "Show the board (markers 0-3). Pop targets with your finger! "
            "[SPACE] reset  [ESC] quit",
            font_size=10, x=10, y=10, anchor_x="left", anchor_y="bottom",
            color=(255, 255, 255, 200),
        )

        pyglet.clock.schedule_interval(lambda dt: self.window.dispatch_event('on_draw'), 1/30.0)
        self.window.push_handlers(on_draw=self.on_draw,
                                  on_key_press=self.on_key_press)

    @staticmethod
    def _display_size(width, height, max_width=1280):
        """Scale (width, height) down so the window is at most ``max_width`` wide
        while keeping the aspect ratio."""
        if width <= max_width:
            return width, height
        scale = max_width / width
        return int(width * scale), int(height * scale)

    def process_frame(self):
        """Grab a webcam frame, extract the board ROI and run game logic.

        Returns the BGR image to display (either the warped board or, while the
        board is not fully visible, the raw frame).
        """
        ret, frame = self.cap.read()
        if not ret:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        markers, _, _ = aruco_board.detect_markers(self.detector, frame)
        quad = aruco_board.get_board_quad(markers)

        if quad is None:
            # Board not (fully) visible: show the raw frame as a hint.
            return frame

        warped, _ = aruco_board.warp_board(frame, quad, self.width, self.height)
        self.finger = self.tracker.track(warped)

        self.target.age += 1
        if self.target.is_hit(self.finger):
            self.score += 1
            self.target.respawn(self.width, self.height)
        elif self.target.age > TARGET_LIFETIME:
            self.target.respawn(self.width, self.height)

        self.draw_game_overlay(warped)
        return warped

    def draw_game_overlay(self, img):
        """Draw the target (red circle) and finger (green crosshair) onto the
        board image."""
        cv2.circle(img, (self.target.x, self.target.y),
                   self.target.radius, (0, 0, 255), -1)
        cv2.circle(img, (self.target.x, self.target.y),
                   self.target.radius, (255, 255, 255), 2)

        if self.finger is not None:
            cv2.drawMarker(img, self.finger, (0, 255, 0),
                           cv2.MARKER_CROSS, 30, 3)

    def on_draw(self):
        self.window.clear()
        board = self.process_frame()
        cv2glet(board).blit(0, 0, 0,
                            width=self.window.width, height=self.window.height)
        self.score_label.text = f"Score: {self.score}"
        self.score_label.draw()
        self.info_label.draw()

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.SPACE:
            self.score = 0
            self.target.respawn(self.width, self.height)
        elif symbol == pyglet.window.key.ESCAPE:
            self.cap.release()
            self.window.close()

    def run(self):
        pyglet.app.run()


def main():
    video_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    game = ARGame(video_id)
    game.run()


if __name__ == "__main__":
    main()
