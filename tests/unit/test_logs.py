import datetime
import logging

import pytest

from hoop_guide.logs import LOG_FILE_NAME
from hoop_guide.logs import setup_logging


@pytest.fixture
def restore_root_logger():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    for handler in root.handlers:
        handler.close()
    root.handlers, root.level = handlers, level


@pytest.mark.usefixtures("restore_root_logger")
def test_setup_logging_creates_run_folder_named_by_start_time(tmp_path):
    started = datetime.datetime(2026, 10, 26, 14, 5, 9)

    run_dir = setup_logging(tmp_path, now=started)

    assert run_dir == tmp_path / "2026-10-26_14-05-09"
    assert run_dir.is_dir()


@pytest.mark.usefixtures("restore_root_logger")
def test_setup_logging_writes_messages_to_log_file(tmp_path):
    run_dir = setup_logging(tmp_path)

    logging.getLogger("hoop_guide.test").info("Robot initialized.")

    log = (run_dir / LOG_FILE_NAME).read_text()
    assert "INFO hoop_guide.test: Robot initialized." in log
