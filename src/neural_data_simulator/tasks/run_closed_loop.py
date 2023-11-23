"""Run all components of the BCI closed loop."""
import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from neural_data_simulator.core.settings import LogLevel
from neural_data_simulator.util.runtime import configure_logger
from neural_data_simulator.util.runtime import initialize_logger
from neural_data_simulator.util.settings_loader import check_config_override_str

SCRIPT_NAME = "nds-run-closed-loop"
logger = logging.getLogger(__name__)

# This is the pipe message indicating that the main task has finished
MAIN_TASK_FINISHED_MSG = "main_task_finished"


def _run_process(args) -> subprocess.Popen:
    logger.info(f"Starting process: {args}")
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
    parser.add_argument(
        "--nds-overrides",
        "-no",
        nargs="*",
        type=check_config_override_str,
        help=(
            "Specify NDS config overrides as key-value pairs, separated by spaces. "
            "For example: -no log_level=DEBUG streamer.lsl_chunk_frequency=50"
        ),
    )
    parser.add_argument(
        "--decoder-overrides",
        "-do",
        nargs="*",
        type=check_config_override_str,
        help=(
            "Specify decoder config overrides as key-value pairs, separated by spaces. "
            "For example: -do log_level=DEBUG decoder.spike_threshold=-210"
        ),
    )
    parser.add_argument(
        "--task-overrides",
        "-to",
        nargs="*",
        type=check_config_override_str,
        help=(
            "Specify task config overrides as key-value pairs, separated by spaces. "
            "For example: -to log_level=DEBUG center_out_reach.task.target_radius=0.03"
        ),
    )

    parser.add_argument(
        "--remote-task",
        action="store_true",
        help="Run without center_out_reach task, expecting a remote task to be used.",
    )

    args = parser.parse_args()

    return args


def _build_param_from_arg(arg_value, param_name):
    if arg_value is None:
        return []
    if isinstance(arg_value, list):
        # Pass in each arg_value element as its string version
        return [param_name] + [str(val) for val in arg_value]
    return [param_name, str(arg_value)]


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
    OVERRIDES_PARAM = "--overrides"

    nds_params = _build_param_from_arg(args.nds_settings_path, SETTINGS_PATH_PARAM)
    nds_params += _build_param_from_arg(args.nds_overrides, OVERRIDES_PARAM)
    decoder_params = _build_param_from_arg(
        args.decoder_settings_path, SETTINGS_PATH_PARAM
    )
    decoder_params += _build_param_from_arg(args.decoder_overrides, OVERRIDES_PARAM)
    task_params = _build_param_from_arg(args.task_settings_path, SETTINGS_PATH_PARAM)
    task_params += _build_param_from_arg(args.task_overrides, OVERRIDES_PARAM)

    logger.info("Starting modules")

    encoder = _run_process(["encoder"] + nds_params)
    ephys = _run_process(["ephys_generator"] + nds_params)
    decoder = _run_process(["decoder"] + decoder_params)

    if args.remote_task:
        logger.info("Running with remote task. Press CTRL+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("CTRL+C received. Exiting...")
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            control_file_path = os.path.join(temp_dir, "center_out_reach_control_file")
            Path(control_file_path).touch(exist_ok=False)
            center_out_reach = _run_process(
                ["center_out_reach", "--control-file", control_file_path] + task_params
            )
            logger.info("Modules started")

            try:
                _wait_for_center_out_reach_main_task(
                    control_file_path, center_out_reach
                )
            except KeyboardInterrupt:
                logger.info("CTRL+C received. Exiting...")
                _terminate_process("center_out_reach", center_out_reach)

            if center_out_reach.poll() is None:
                logger.info("Waiting for center_out_reach")
                center_out_reach.wait()

    _terminate_process("encoder", encoder)
    _terminate_process("ephys_generator", ephys)
    _terminate_process("decoder", decoder)

    logger.info("Done")


if __name__ == "__main__":
    run()
