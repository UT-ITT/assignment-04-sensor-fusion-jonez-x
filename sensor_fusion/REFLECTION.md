# Sensor Fusion - Reflection

## Implementation

The program tracks an ArUco board (markers 0-3) via the webcam, extracts and
perspectively warps the region between the markers, and tracks a moving marker
(id 5) inside that rectangle. Its position is projected into board space using
the same perspective matrix and shown as a **red dot**.

In parallel, accelerometer data is read from a mobile device via DIPPID. The
raw acceleration is scaled and integrated twice over time (acceleration ->
velocity -> position) by `AccelerometerIntegrator`. A **complementary filter**
fuses the two sources:

```
prediction = alpha * accel_estimate + (1 - alpha) * camera_position
```

The fused result is drawn as a **green dot**. The arrow keys adjust `alpha` at
runtime and DIPPID Button 1 resets the integrator to the current camera
position.

## How different alpha values affect the prediction

- **alpha close to 0** — The prediction follows the camera almost exclusively.
  It is drift-free and accurate over time, but inherits the camera's weaknesses:
  it is laggy, jumps when the marker is briefly lost, and is noisy frame to
  frame. When the marker leaves the frame, the prediction effectively freezes.

- **alpha close to 1** — The prediction follows the integrated accelerometer.
  It reacts instantly and stays smooth even when the camera loses the marker,
  but double-integrating a noisy accelerometer accumulates **drift**: the green
  dot slowly wanders away from the true position and must be reset with Button 1.

- **alpha around 0.3-0.6** — The sweet spot. The accelerometer provides fast,
  smooth short-term motion while the camera continuously corrects long-term
  drift. The green dot feels responsive yet stays anchored to reality.

The key trade-off is **responsiveness vs. drift**: the accelerometer is fast but
drifts, the camera is slow/noisy but absolute. The complementary filter lets us
pick where on that spectrum we want to sit. We also added a small soft
correction that nudges the integrator back toward the camera position, which
keeps even high-alpha runs from drifting off-screen between manual resets.
