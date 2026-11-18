# Manual

How to run the application and why it behaves the way it does.

## Running

```sh
uv run main.py [--real] [--robot crs93|crs97] [--no-home] [--no-view]
```

| Command | What it does | Why |
|---|---|---|
| `uv run main.py` | Starts with a virtual CRS93. No connection to the controller, only computations like `fk` and `ik` work, calls that need the robot are skipped. | Safe default. Nobody connects to the real robot by accident. |
| `uv run main.py --real` | Connects to the real robot and shows a warning. | Connecting to the real robot must be an explicit decision. |
| `uv run main.py --real --robot crs97` | Connects to the real CRS97. Default is `crs93`. | Both stations are in the lab and `ctu_crs` supports both. |
| `uv run main.py --real --no-home` | Initializes the robot without hard and soft home. | Homing is only needed after the control unit was powered off. After an emergency stop or opening the cage it is not, so we save time and unnecessary motion. |
| `uv run main.py --real --no-view` | Runs without the camera window. The camera and the vision keep running. | The window needs a display, e.g. over SSH there is none. |

## Camera window

With the real robot a window with the live camera image opens at start. The virtual robot has no camera, so there is no window.

The window shows:

- Every detected ArUco marker (`DICT_4X4_50`) with its outline, ID and a red dot in its top left corner, also markers that do not belong to the plate. This way the IDs on the plates can be checked.
- A status line in the top left corner, e.g. `10.0 fps | markers: 0, 1`. The frame rate is averaged over the last 10 frames and shows how many images the application really gets.

The plate frame, rod and prediction will be added when they are implemented.

The window is display only. Keys pressed in it do nothing, all control stays in the terminal. Closing it does not stop the robot or the application, the window does not come back until the next start.

If the camera fails, the image freezes and the error is in `app.log`, the robot is not affected.

## Startup

1. With `--real` a warning shows that the application is connected to the real robot.
2. The initialization starts only after pressing Enter, so the robot does not move right after the command is run. While it runs, the terminal shows that the robot is being initialized.
3. The main menu appears.

With the virtual robot the initialization is skipped and the menu appears right away.

The header above the menu always shows the robot, whether it is real or virtual, the running mode, the current state and the folder with logs.

## Logs

Every run creates a folder `runs/<date_time>/` with `app.log`. Everything the application does is logged there: start, warnings, initialization, menu choices, errors with details and shutdown. Telemetry and camera images of the run will be stored in the same folder. Error messages in the terminal point to this file.

## Menu

| Option | What it does | Why |
|---|---|---|
| A1, A2, B, C, D | Runs the puzzle mode. Not implemented yet. | One mode per puzzle variant, from the easiest to the hardest. |
| Soft home | Moves the robot to the home position. | Return to a known position, e.g. after stopping a mode. |
| Release | Engages the brakes and disconnects feedback. Wait until the robot stops and do not use it above the work surface, the arm can slightly drop. | Required before entering the safety cage. |
| Reset motors | Resets motors in an error state. | Recovery after a motor error (flashing green and red LED). |
| Manual move | Moves the hoop from the keyboard and records calibration points, see below. Only with the real robot. | Calibration of the camera without preparing poses in advance. |
| Quit | Same as Ctrl+C. | |

Soft home, Release and Reset motors are blocked with the virtual robot, they need the controller.

## Manual move

The table shows the target and the current position of the hoop center in the robot frame, the target and the current rotation of the hoop, the number of recorded points and what happened last.

| Key | What it does |
|---|---|
| Up, Down | Selects the row x, y, z, rotation x, rotation y, rotation z or Exit. |
| Left, Right | Changes the target position in the selected axis by 10 mm or the selected target rotation by 5°, Right increases it. |
| Enter | On a position or rotation row moves the hoop to the whole target. On Exit returns to the menu. |
| Space | Records a calibration point: the hoop center in the newest camera image and the current position. |

- The robot moves only after Enter. The target starts at the pose the hoop had when the mode started.
- The rotations are angles about the x, y and z axis of the robot frame, applied in this order. With rotation y at 0° the flange points up, at 90° forward along x as in the home position, at 180° down.
- The same orientation can be written with other angles, so the current rotation can show other numbers than the target, e.g. 180°, 0°, 180° for the target 0°, 180°, 0°.
- While the robot moves, keys are ignored. The move can only be stopped by the Emergency Stop button.
- A point is refused when the camera has no image yet, the hoop is not found, or the height differs from the first point by more than 10 mm.
- After 9 points the homography is computed. It is saved to `config/homography.json` only when its error on the 9 points is at most 10 mm, otherwise the file stays as it was. In both cases the points are dropped and counting starts again.
- The camera window uses the new homography after the application is restarted.
- Leaving the mode drops the recorded points.

## Keys

| Key | What it does | Why |
|---|---|---|
| `c` | Stops the running mode. No new commands are sent, the robot finishes the current short step and stays where it is. Back to the menu. | Uses only the public `ctu_crs` API. An immediate stop through the controller is not available there. |
| Ctrl+C | Stops the motion, waits until the robot stands still, then `soft_home()` and `close()`. Further Ctrl+C during this is ignored. Works anytime. | The robot always ends in a known state and the connection is closed, as the station manual requires. |

The software stop does not replace the Emergency Stop button. If the robot behaves unexpectedly, press it.
