"""Logging setup, every run gets its own folder."""

import datetime
import logging
from pathlib import Path

RUNS_DIR = Path("runs")
LOG_FILE_NAME = "app.log"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(
    runs_dir: Path = RUNS_DIR, now: datetime.datetime | None = None
) -> Path:
    """Creates a folder for this run and logs into its app.log.

    Args:
        runs_dir: Folder with all runs.
        now: Start time of the run, current time if None.

    Returns:
        Folder of this run, telemetry and images of the run belong here too.
    """
    started = now or datetime.datetime.now()
    run_dir = runs_dir / started.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=run_dir / LOG_FILE_NAME,
        level=logging.INFO,
        format=LOG_FORMAT,
        force=True,
    )
    return run_dir
