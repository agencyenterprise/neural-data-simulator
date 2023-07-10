"""Run all components of the BCI closed loop."""
import argparse
import logging
from pathlib import Path
import subprocess
import tempfile
import time

from neural_data_simulator.settings import LogLevel
from neural_data_simulator.util.runtime import configure_logger
from neural_data_simulator.util.runtime import initialize_logger

SCRIPT_NAME = "nds-run-closed-loop"
logger = logging.getLogger(__name__)

# This is the pipe message indicating that the main task has finished
MAIN_TASK_FINISHED_MSG = "main_task_finished"


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


def _wait_for_center_out_reach_to_finish(control_file, center_out_reach):
    # TODO: explain what are we checking here
    logger.info("Waiting for center out reach task")
    while True:
        line = str(control_file.readline())
        if MAIN_TASK_FINISHED_MSG in line:
            logger.info("Center out reach task finished")
            break
        if center_out_reach.poll() is not None:
            logger.info("Center out reach task stopped unexpectedly.")
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

    nds_params = _build_param_from_arg(args.nds_settings_path, "--settings-path")
    decoder_params = _build_param_from_arg(
        args.decoder_settings_path, "--settings-path"
    )
    task_params = _build_param_from_arg(args.task_settings_path, "--settings-path")

    logger.info("Starting modules")
    encoder = subprocess.Popen(["encoder"] + nds_params)
    ephys = subprocess.Popen(["ephys_generator"] + nds_params)
    decoder = subprocess.Popen(["decoder"] + decoder_params)

    with tempfile.NamedTemporaryFile() as control_file:
        center_out_reach = subprocess.Popen(
            ["center_out_reach", "--control-file", control_file.name] + task_params
        )
        try:
            _wait_for_center_out_reach_to_finish(control_file, center_out_reach)
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
