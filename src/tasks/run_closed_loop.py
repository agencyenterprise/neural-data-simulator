"""Run all components of the BCI closed loop."""
import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from neural_data_simulator.settings import LogLevel
from neural_data_simulator.util.runtime import configure_logger
from neural_data_simulator.util.runtime import initialize_logger

SCRIPT_NAME = "nds-run-closed-loop"
logger = logging.getLogger(__name__)

# This is the pipe message indicating that the main task has finished
MAIN_TASK_FINISHED_MSG = "main_task_finished"


def _run_process(args) -> subprocess.Popen:
    return subprocess.Popen(args, shell=sys.platform == "win32")


def _terminate_process(label: str, popen_process: subprocess.Popen, timeout: int = 5):
    logger.info(f"Terminating {label}")
    popen_process.terminate()
    try:
        popen_process.wait(timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        logger.warning(f"{label} did not terminate in time. Killing it")
        popen_process.kill()


def _parse_args():
    parser = argparse.ArgumentParser(description="Run closed loop.")
    parser.add_argument(
        "--nds-settings-path",
        type=Path,
        help="Path to the yaml file containing the NDS config.",
    )
    parser.add_argument(
        "--decoder-settings-path",
        type=Path,
        help="Path to the yaml file containing the decoder config.",
    )
    parser.add_argument(
        "--task-settings-path",
        type=Path,
        help="Path to the yaml file containing the task config.",
    )
    return parser.parse_args()


def _build_param_from_arg(arg_value, param_name):
    params = []
    if arg_value is not None:
        params = [param_name, str(arg_value)]
    return params


def _wait_for_center_out_reach_main_task(
    control_file_path: str, center_out_reach: subprocess.Popen
):
    """Waits for the center_out_reach main task to finish.

    This function is blocking and will only return when the center_out_reach
    has sent a message indicating that his main task has finished or when
    the center_out_reach process has stopped unexpectedly, indicating that the
    main task was also terminated.

    Args:
        control_file_path (str): Path to the control file.
        center_out_reach (subprocess.Popen): The center_out_reach process.
    """
    logger.info("Waiting for center_out_reach main task")
    with open(control_file_path, "r") as file:
        while True:
            line = str(file.readline())
            if MAIN_TASK_FINISHED_MSG in line:
                logger.info("center_out_reach main task finished")
                break
            if center_out_reach.poll() is not None:
                logger.info("center_out_reach stopped unexpectedly.")
                break
            time.sleep(0.5)


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

    args = _parse_args()

    SETTINGS_PATH_PARAM = "--settings-path"

    nds_params = _build_param_from_arg(args.nds_settings_path, SETTINGS_PATH_PARAM)
    decoder_params = _build_param_from_arg(
        args.decoder_settings_path, SETTINGS_PATH_PARAM
    )
    task_params = _build_param_from_arg(args.task_settings_path, SETTINGS_PATH_PARAM)

    logger.info("Starting modules")

    encoder = _run_process(["encoder"] + nds_params)
    ephys = _run_process(["ephys_generator"] + nds_params)
    decoder = _run_process(["decoder"] + decoder_params)

    with tempfile.TemporaryDirectory() as temp_dir:
        control_file_path = os.path.join(temp_dir, "center_out_reach_control_file")
        Path(control_file_path).touch(exist_ok=False)
        center_out_reach = _run_process(
            ["center_out_reach", "--control-file", control_file_path] + task_params
        )
        logger.info("Modules started")

        try:
            _wait_for_center_out_reach_main_task(control_file_path, center_out_reach)
        except KeyboardInterrupt:
            logger.info("CTRL+C received. Exiting...")
            _terminate_process("center_out_reach", center_out_reach)

        _terminate_process("encoder", encoder)
        _terminate_process("ephys_generator", ephys)
        _terminate_process("decoder", decoder)

        if center_out_reach.poll() is None:
            logger.info("Waiting for center_out_reach")
            center_out_reach.wait()

    logger.info("Done")


if __name__ == "__main__":
    run()
