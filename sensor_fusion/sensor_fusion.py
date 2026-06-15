"""
sensor_fusion.py - Sensor Fusion (Assignment 4, Task 3)

Reads the webcam image and tracks a board with four ArUco corner markers (ids
0-3), extracting and warping the region between them to a rectangle (same as
Task 2). A moving ArUco marker with id 5 (e.g. attached to the back of a
smartphone) is tracked inside that rectangle and shown as a RED dot.

Additionally, accelerometer data is read from a mobile device via DIPPID and
integrated over time. A complementary filter fuses the camera position with the
accelerometer-based position estimate. The fused prediction is shown as a GREEN
dot.

Controls:
    LEFT / RIGHT arrow - decrease / increase the filter weight alpha
    UP / DOWN arrow    - (same as RIGHT / LEFT, alternative binding)
    DIPPID Button 1    - reset the prediction to the current camera position
    ESC                - quit

The complementary filter
------------------------
    prediction = alpha * accel_estimate + (1 - alpha) * camera_position

A high alpha trusts the (fast but drift-prone) accelerometer more; a low alpha
trusts the (slower, noisier, but drift-free) camera more. See REFLECTION.md.

Usage:
    python sensor_fusion.py [video_id] [dippid_port]
"""

import sys

import cv2
import numpy as np
import pyglet

import aruco_board
from DIPPID import SensorUDP

# ArUco marker attached to the tracked device (smartphone).
MOVING_MARKER_ID = 5
DEFAULT_DIPPID_PORT = 5700

# Raw accelerometer values are tiny compared to pixel space, so scale them up
# before integration (see assignment hint).
ACCEL_SCALE = 4000.0

INITIAL_ALPHA = 0.5
ALPHA_STEP = 0.05


class AccelerometerIntegrator:
    """Integrates accelerometer data over time into a position estimate.

    The accelerometer gives acceleration; integrating once yields velocity and
    integrating again yields position. We keep a velocity term and a position
    offset, both of which can be reset.
    """

    def __init__(self):
        self.velocity = np.zeros(2, dtype="float64")
        self.position = np.zeros(2, dtype="float64")

    def reset(self, position):
        """Reset the integrator to a known position with zero velocity."""
        self.position = np.array(position, dtype="float64")
        self.velocity = np.zeros(2, dtype="float64")

    def update(self, accel_xy, dt):
        """Advance the integration by ``dt`` seconds using ``accel_xy``.

        ``accel_xy`` is the (already scaled) acceleration in board-pixel units.
        Returns the current integrated position.
        """
        accel = np.array(accel_xy, dtype="float64")
        # Euler integration with light damping; the damping stops the double
        # integration from running away while the device is roughly still.
        self.velocity += accel * dt
        self.velocity *= 0.9
        self.position += self.velocity * dt
        return self.position.copy()


class SensorFusion:
    """Main application fusing camera marker tracking and accelerometer data."""

    def __init__(self, video_id=0, dippid_port=DEFAULT_DIPPID_PORT):
        self.cap = cv2.VideoCapture(video_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam {video_id}")

        # Request a moderate capture resolution for good performance.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        self.detector = aruco_board.create_detector()

        self.sensor = SensorUDP(dippid_port)
        self.sensor.register_callback("button_1", self.on_button_1)

        self.integrator = AccelerometerIntegrator()
        self.alpha = INITIAL_ALPHA

        # Positions in board pixel coordinates.
        self.camera_position = np.array([self.width / 2, self.height / 2])
        self.prediction = self.camera_position.copy()
        self.integrator.reset(self.camera_position)

        # Set by the button callback (running on the receive thread), consumed in
        # the update loop on the main thread.
        self._reset_requested = False

        win_w, win_h = self._display_size(self.width, self.height)
        self.window = pyglet.window.Window(win_w, win_h,
                                           caption="Sensor Fusion")
        self.info_label = pyglet.text.Label(
            "", font_size=12, x=10, y=win_h - 10,
            anchor_x="left", anchor_y="top", color=(255, 255, 255, 255),
        )

        # Drive the simulation at a fixed rate so dt is well defined, and force a
        # redraw at ~30 fps (on_draw is otherwise only called on window events).
        pyglet.clock.schedule_interval(self.update, 1 / 60.0)
        pyglet.clock.schedule_interval(
            lambda dt: self.window.dispatch_event('on_draw'), 1 / 30.0)
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

    # ----- DIPPID -----------------------------------------------------------
    def on_button_1(self, value):
        """DIPPID button 1 callback: request a prediction reset."""
        if int(value) == 1:
            self._reset_requested = True

    def read_acceleration(self):
        """Read scaled (x, y) acceleration from DIPPID, or zeros if unavailable.

        We map the device's x/y accelerometer axes onto the board's x/y axes.
        """
        accel = self.sensor.get_value("accelerometer")
        if not accel:
            return np.zeros(2)
        ax = float(accel.get("x", 0.0)) * ACCEL_SCALE
        ay = float(accel.get("y", 0.0)) * ACCEL_SCALE
        return np.array([ax, ay])

    # ----- Tracking ---------------------------------------------------------
    def update_camera_position(self):
        """Grab a webcam frame and update the camera-based marker position.

        Returns ``(image, board_detected)``. When the board is fully visible the
        image is the warped board rectangle; otherwise it is the raw frame with
        any detected markers drawn on it (so the user gets visual feedback while
        positioning the board).
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, False

        markers, corners, ids = aruco_board.detect_markers(self.detector, frame)
        quad = aruco_board.get_board_quad(markers)
        if quad is None:
            # Board not fully visible: show the raw frame with detected markers.
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            return frame, False

        warped, matrix = aruco_board.warp_board(
            frame, quad, self.width, self.height
        )

        # Project the moving marker's center into warped board space using the
        # same perspective matrix.
        if MOVING_MARKER_ID in markers:
            center = aruco_board.marker_center(markers[MOVING_MARKER_ID])
            point = np.array([[center]], dtype="float32")
            mapped = cv2.perspectiveTransform(point, matrix)[0][0]
            self.camera_position = np.array([mapped[0], mapped[1]])

        return warped, True

    # ----- Fusion -----------------------------------------------------------
    def update(self, dt):
        """Fixed-timestep update: integrate accelerometer and fuse with camera."""
        if self._reset_requested:
            self.integrator.reset(self.camera_position)
            self.prediction = self.camera_position.copy()
            self._reset_requested = False

        # Integrate accelerometer into an absolute position estimate.
        accel_xy = self.read_acceleration()
        accel_position = self.integrator.update(accel_xy, dt)

        # Complementary filter: blend accelerometer estimate and camera position.
        self.prediction = (self.alpha * accel_position
                           + (1.0 - self.alpha) * self.camera_position)

        # Keep the integrator anchored near the camera so it cannot drift away
        # indefinitely (soft correction proportional to 1 - alpha).
        correction = (1.0 - self.alpha) * (self.camera_position
                                           - self.integrator.position)
        self.integrator.position += correction * 0.1

    # ----- Rendering --------------------------------------------------------
    def on_draw(self):
        self.window.clear()
        image, board_detected = self.update_camera_position()

        if image is not None:
            if board_detected:
                self.draw_dots(image)
            self.cv2glet(image).blit(0, 0, 0,
                                     width=self.window.width,
                                     height=self.window.height)

        status = "" if board_detected else "   [board (markers 0-3) not fully visible]"
        self.info_label.text = (
            f"alpha={self.alpha:.2f}  (LEFT/RIGHT to adjust, "
            f"DIPPID Button 1 to reset)   red=camera  green=prediction{status}"
        )
        self.info_label.draw()

    def draw_dots(self, img):
        """Draw the camera position (red) and the fused prediction (green)."""
        cam = (int(self.camera_position[0]), int(self.camera_position[1]))
        pred = (int(self.prediction[0]), int(self.prediction[1]))
        cv2.circle(img, cam, 12, (0, 0, 255), -1)     # red: camera
        cv2.circle(img, pred, 12, (0, 255, 0), -1)    # green: prediction

    def cv2glet(self, img):
        """Convert an OpenCV BGR image to a pyglet image."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rows, cols, channels = img_rgb.shape
        return pyglet.image.ImageData(
            width=cols, height=rows, fmt='RGB',
            data=img_rgb.tobytes(), pitch=-channels * cols,
        )

    # ----- Input ------------------------------------------------------------
    def on_key_press(self, symbol, modifiers):
        key = pyglet.window.key
        if symbol in (key.RIGHT, key.UP):
            self.alpha = min(1.0, self.alpha + ALPHA_STEP)
        elif symbol in (key.LEFT, key.DOWN):
            self.alpha = max(0.0, self.alpha - ALPHA_STEP)
        elif symbol == key.ESCAPE:
            self.shutdown()

    def shutdown(self):
        self.cap.release()
        self.sensor.disconnect()
        self.window.close()

    def run(self):
        pyglet.app.run()


def main():
    video_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    dippid_port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DIPPID_PORT
    app = SensorFusion(video_id, dippid_port)
    app.run()


if __name__ == "__main__":
    main()
