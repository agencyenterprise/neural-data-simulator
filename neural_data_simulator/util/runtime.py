"""Functions commonly used by scripts."""
import contextlib
import logging
from logging.handlers import MemoryHandler
import os
from pathlib import Path
import sys
from typing import Any, Optional, Union

from neural_data_simulator.inputs import Input
from neural_data_simulator.outputs import Output
from neural_data_simulator.settings import LogLevel

logger = logging.getLogger(__name__)

NDS_HOME = os.path.join(os.path.expanduser("~"), ".nds")


def initialize_logger(script_name: str):
    """Initialize the logger.

    Store log messages in memory until the logger is configured.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.NOTSET)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_get_log_message_format(script_name)))
    temp_handler = MemoryHandler(1000, target=handler)
    root_logger.addHandler(temp_handler)


def configure_logger(script_name: str, log_level: LogLevel):
    """Set up the logger."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, MemoryHandler):
            level = logging._nameToLevel[log_level.value.upper()]
            filtered_buffer = filter(
                lambda record: record.levelno >= level, handler.buffer
            )
            handler.buffer = list(filtered_buffer)
            break

    logging.basicConfig(
        format=_get_log_message_format(script_name), level=log_level.value, force=True
    )
    os.environ["LSLAPICFG"] = os.path.join(NDS_HOME, "lsl.config")


def _get_log_message_format(script_name: str):
    return f"%(levelname)s [{script_name}]: %(message)s"


def get_abs_path(
    abs_or_relative_path: Union[str, Path], relative_to: str = os.getcwd()
) -> str:
    """Return the absolute path for a resource.

    If an absolute path is passed as the first parameter then
    this is returned unchanged and the second parameter is ignored.
    If a relative path is passed as the first parameter then
    the second parameter is considered a sibling resource and
    the function returns an absolute path where the first parameter
    is relative to the path of the second parameter.

    Args:
        abs_or_relative_path: An absolute or relative path.
        relative_to: A file or directory absolute path.
          Defaults to current working directory.

    Returns:
        The absolute path of the first parameter.
    """
    if os.path.isabs(abs_or_relative_path):
        return str(abs_or_relative_path)

    parent_dir = relative_to
    if os.path.isfile(parent_dir):
        parent_dir = os.path.dirname(relative_to)

    full_path = os.path.join(parent_dir, abs_or_relative_path)
    if Path(full_path).exists():
        return full_path
    else:
        return os.path.join(NDS_HOME, abs_or_relative_path)


def get_sample_data_dir() -> str:
    """Get the path for the sample data directory."""
    return os.path.join(NDS_HOME, "sample_data")


def get_plugins_dir() -> str:
    """Get the path for the plugins directory."""
    return os.path.join(NDS_HOME, "plugins")


def get_configs_dir() -> str:
    """Get the path for the directory containing the configuration files."""
    return NDS_HOME


def unwrap(o: Optional[Any]) -> Any:
    """Unwraps the given optional.

    Args:
        o: The optional to unwrap.

    Returns:
        The wrapped value that is different from None.

    Raises:
        ValueError: If the optional is None.
    """
    if not o:
        raise ValueError("Tried to unwrap None value")
    return o


@contextlib.contextmanager
def open_connection(io: Union[Optional[Input], Optional[Output]]):
    """Open a managed connection to the given input or output.

    The connection is released after it is consumed.

    Args:
        io: An optional input or output to connect to. If None
            yield without doing anything.
    """
    if not io:
        yield
    else:
        try:
            io.connect()
            yield
        finally:
            io.disconnect()
