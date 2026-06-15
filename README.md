[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/AktWbCri)
# assignment-04-CV-Sensor-Fusion

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Task 1 - Perspective Transformation

```bash
cd perspective_transformation
python image_extractor.py --input sample_image.jpg --output result.jpg --width 800 --height 600
```

Left-click four points to select a region. **ESC** discards and restarts,
**S** saves the warped result, **Q** quits.

## Task 2 - AR Game

```bash
cd ar_game
python AR_game.py [video_id]
```

Show a board with ArUco markers (ids 0-3, dictionary `DICT_6X6_250`) in the four
corners. The region between the markers is warped to the webcam resolution and
shown in a pyglet window. Pop the red targets with your finger.
**SPACE** resets the score, **ESC** quits.

## Task 3 - Sensor Fusion

```bash
cd sensor_fusion
python sensor_fusion.py [video_id] [dippid_port]   # dippid_port default: 5700
```

Same board as Task 2. A moving ArUco marker (id 5) is tracked as a **red dot**.
Accelerometer data from the DIPPID app is integrated and fused with the camera
position via a complementary filter, shown as a **green dot**.
**LEFT/RIGHT** (or DOWN/UP) adjust the filter weight alpha, **DIPPID Button 1**
resets the prediction, **ESC** quits. See `REFLECTION.md` for the write-up.