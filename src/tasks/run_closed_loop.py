"""Run all components of the BCI closed loop."""
import logging
import os
import subprocess
import tempfile

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


def _create_pipe_file(pipe_name: str) -> tuple[str, str]:
    temp_dir = tempfile.mkdtemp()
    pipe_path = os.path.join(temp_dir, pipe_name)
    os.mkfifo(pipe_path)
    return (temp_dir, pipe_path)


def _wait_for_pipe_message(pipe_path: str, pipe_message: str) -> bool:
    """
    Wait for the specified pipe_message in the specified pipe_path file.
    Returns True if the message was found, False if the process was interrupted.
    """
    try:
        with open(pipe_path, "r") as center_out_reach_pipe:
            for line in center_out_reach_pipe:
                if pipe_message in line:
                    logger.info("Main task finished")
                    break
    except KeyboardInterrupt:
        logger.info("CTRL+C received. Exiting...")
        return False
    return True


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

    temp_dir, center_out_reach_pipe_path = _create_pipe_file("center_out_reach")
    center_out_reach = subprocess.Popen(
        ["center_out_reach", "--pipe", center_out_reach_pipe_path]
    )
    logger.info("Modules started")

    main_task_finished = _wait_for_pipe_message(
        center_out_reach_pipe_path,
        MAIN_TASK_FINISHED_PIPE_MSG,
    )
    _terminate_process("encoder", encoder)
    _terminate_process("ephys_generator", ephys)
    _terminate_process("decoder", decoder)
    if not main_task_finished:
        _terminate_process("center_out_reach", center_out_reach)
    else:
        logger.info("Waiting for center_out_reach")
        center_out_reach.wait()

    os.remove(center_out_reach_pipe_path)
    os.rmdir(temp_dir)
    logger.info("Done")


if __name__ == "__main__":
    run()
