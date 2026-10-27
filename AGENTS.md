# AGENTS.md

Project context is in `README.md`.

## Stack

- Python 3.12, dependencies managed by `uv` only (`uv add`, `uv run`), never `pip`.
- `numpy` for math, `ctu_crs` for the robot (`CRS93`) and the camera.
- Do not add new dependencies without asking.

## Project structure

Target layout. Create a folder or module only when there is code for it.

```
main.py                  # entry point, only wires the modules together
hoop_guide/              # main package
├── app.py               # application flow: startup, menu, shutdown
├── cli.py               # command line arguments
├── ui.py                # terminal UI built on rich
├── keyboard.py          # stop key c while a mode runs
├── logs.py              # logging, one folder per run in runs/
├── types.py             # Pose, JointConfig, ImagePoint, Homography
├── geometry.py          # transforms, rotation about an axis
├── rods.py              # rod models, centerlines from the drawings
├── vision.py            # ArUco detection, homography
├── calibration.py       # calibration from point pairs
├── estimation.py        # center of rotation, ω, phase, pose prediction
├── planning.py          # hoop path along the rod, handle orientation
├── control.py           # tracking loop, step timing
└── hardware/            # the only place that uses ctu_crs
    ├── interfaces.py    # Protocols for robot and camera
    ├── robot.py         # thin wrapper over ctu_crs, real or virtual robot
    └── fake.py          # fake robot and camera for tests and simulation
tests/
├── conftest.py          # shared fixtures (rod models, fake robot, sample poses)
├── unit/                # mirrors hoop_guide/ 1:1, one test file per module
│   ├── test_geometry.py
│   ├── test_estimation.py
│   └── hardware/
│       └── test_fake.py
├── integration/         # whole pipeline with fake robot and camera
│   └── test_puzzle_a.py
└── lab/                 # needs the real robot, run manually in the lab
    └── test_calibration.py
config/                  # calibration and lab settings, committed
```

- Start with flat modules, split one into a subpackage only when it grows.
- `app.py` uses the robot and the UI only through Protocols, so it is tested with fakes. `ui.py` and `hardware/robot.py` are excluded from coverage and checked in the lab.

## Language

Everything in the repository is in English: code, comments, docstrings, terminal output, logs, documentation and commit messages.

## Code quality

- Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html), enforced by `ruff` (config in `pyproject.toml`).
- Type hints on all functions, Google style docstrings on public modules, classes and functions.
  - do not add unnecessary or duplicate doc blocks/comments
  - add a comment only when the code itself cannot tell it. Together with the code it must be enough to understand what is going on, as short as possible but not shorter
- Code is type checked by `mypy --strict`. Domain quantities have their own types (`typing.NewType`), never a bare `np.ndarray`, e.g. `Pose` (4x4 transform), `JointConfig` (6 joint angles), `ImagePoint`, `Homography`. Shape and frame are documented once at the type definition.
- Boolean variables, attributes, arguments and properties start with `is_`, `has_`, `should_` or `can_`, e.g. `is_real`, `should_home`, never `real` or `home`. Ruff cannot check this, keep it in review.
- Units in SI (meters, radians, seconds). Note the unit in the name or docstring where it is not obvious.
- Do not overcomplicate things if there is better, easier and more readable solution to problem.

## Design principles

- Separation of concerns: vision, motion estimation, trajectory planning and robot control are separate modules with clear interfaces.
- Single responsibility: one module, class or function does one thing. Prefer small pure functions for math.
- OOP where there is state (robot, camera, estimator), plain functions for stateless computation.
- Composition over inheritance. Depend on small interfaces (e.g. `typing.Protocol`), so real hardware can be swapped for a fake in tests and simulation.
- KISS and YAGNI: solve the current problem, do not build for hypothetical future needs.
- DRY: shared logic lives in one place, but do not merge code that only looks similar.
- Use design patterns only when they solve a real problem here, not for their own sake.
- Fail fast: validate inputs at module boundaries and raise clear exceptions, never fail silently.
- No magic numbers: named constants for dimensions, limits and tolerances.

## Testing

- Tests in `tests/` with `pytest`, file names `test_<module>.py`.
  - `unit/` mirrors `hoop_guide/` 1:1: `hoop_guide/vision.py` is tested in `tests/unit/test_vision.py`.
  - `integration/` tests several modules together, always with fake hardware.
  - `lab/` needs the real robot and camera. It is excluded from `make` and the pre-commit hook and run manually with `uv run pytest tests/lab --no-cov`.
- Every new function with logic gets a test. Geometry and prediction code must be testable without the robot or camera.
- Hardware code (anything talking to `CRS93` or the camera) stays in a thin layer, so the rest can be tested offline.
- Test behavior, not implementation. A refactor that keeps behavior must not break tests.
- Tests are fast, independent and repeatable: no order dependency, no shared mutable state, fixed seeds for random data.
- Structure each test as Arrange, Act, Assert. One behavior per test.
- Names describe scenario and expected result, e.g. `test_predict_pose_after_half_turn_is_mirrored`.
- Use `@pytest.mark.parametrize` (with `ids`) instead of copying tests, and fixtures in `conftest.py` for shared setup.
- Compare floats and arrays with `pytest.approx` or `np.testing.assert_allclose`, never `==`.
- Cover edge cases: zero and negative ω, singular and out-of-reach poses, missing markers.
- Use fakes for hardware, avoid mocking internals of our own code.
- When fixing a bug, first write a test that reproduces it.
- Use `hypothesis` property-based tests for geometry and prediction, e.g. `fk(ik(pose)) == pose` or a full turn returns the original pose.
- Branch coverage of `hoop_guide/` must stay at least 90 % (`ui.py` and `hardware/robot.py` are excluded).

## Planned (implementation tasks)

- Safety layer before every `move_to_q`: joint limits, workspace bounds, maximum step between commands and hoop to rod distance from the model. On failure the robot stops and nothing is sent.
- Run recording: images, estimates, sent commands and timestamps, so a run can be replayed offline and turned into a regression test.
- Before `soft_home()` on exit, retract the hoop from the rod along the trajectory, otherwise the direct move home can hit the rod.
- Verify in the lab whether the controller `STOP:` command can be used for an immediate stop on `c`.

## Before finishing a change

```sh
uv run ruff format .
uv run ruff check .
uv run mypy .
uv run pytest
```

All must pass. `make` runs all of them. The same checks run automatically as a git pre-commit hook (`.pre-commit-config.yaml`), never bypass it with `--no-verify`.
