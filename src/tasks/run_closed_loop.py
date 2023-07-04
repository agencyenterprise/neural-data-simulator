"""Run all components of the BCI closed loop."""
import logging
import os
import subprocess
import tempfile
import time

from neural_data_simulator.settings import LogLevel
from neural_data_simulator.util.runtime import configure_logger
from neural_data_simulator.util.runtime import initialize_logger

SCRIPT_NAME = "nds-run-closed-loop"
logger = logging.getLogger(__name__)

# This is the pipe message indicating that the main task has finished
MAIN_TASK_FINISHED_PIPE_MSG = "main_task_finished"


def _terminate_process(label: str, popen_process: subprocess.Popen, timeout: int = 5):
    logger.info(f"Terminating {label}")
    popen_process.terminate()
    try:
        popen_process.wait(timeout)
    except subprocess.TimeoutExpired:
        logger.warning(f"{label} did not terminate in time. Killing it")
        popen_process.kill()


def _block_until_message(file_path: str, message: str) -> bool:
    """Block until a message is received in a file.

    Args:
        file_path: The path to the file to read.
        message:  Message to wait for.

    Returns:
        True if the message was received, False otherwise.
    """
    try:
        with open(file_path, "r") as file:
            while True:
                line = file.readline()
                if message in line:
                    logger.info(f"Message {message} received")
                    return True
                time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("CTRL+C received. Exiting...")
        return False


def run():
    """Start all components."""
    try:
        import pygame  # noqa: F401
        import screeninfo  # noqa: F401
    except ImportError:
        print(
            "Not running closed loop because neural-data-simulator "
            "extras couldn't be imported"
        )
        print("Please reinstall neural-data-simulator with extras by running:")
        print('pip install "neural-data-simulator[extras]"')
        return

    initialize_logger(SCRIPT_NAME)
    configure_logger(SCRIPT_NAME, LogLevel._INFO)

    logger.info("Starting modules")
    encoder = subprocess.Popen(["encoder"])
    ephys = subprocess.Popen(["ephys_generator"])
    decoder = subprocess.Popen(["decoder"])

    with tempfile.TemporaryDirectory() as temp_dir:
        control_file_path = os.path.join(temp_dir, "center_out_reach_control_file")
        open(control_file_path, "x")

        center_out_reach = subprocess.Popen(
            ["center_out_reach", "--control-file", control_file_path]
        )
        logger.info("Modules started")

        message_received = _block_until_message(
            control_file_path, MAIN_TASK_FINISHED_PIPE_MSG
        )

        _terminate_process("encoder", encoder)
        _terminate_process("ephys_generator", ephys)
        _terminate_process("decoder", decoder)

        if not message_received:
            _terminate_process("center_out_reach", center_out_reach)
        else:
            logger.info("Waiting for center_out_reach")
            center_out_reach.wait()

    logger.info("Done")


if __name__ == "__main__":
    run()
