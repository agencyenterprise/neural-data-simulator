"""Run the center-out reach task GUI."""
import argparse
import logging
from pathlib import Path
import re
from typing import cast, Tuple

import numpy as np
from pydantic_yaml import VersionedYamlModel
from tasks.center_out_reach.input_events import InputHandler
from tasks.center_out_reach.metrics import MetricsCollector
from tasks.center_out_reach.scalers import PixelsToMetersConverter
from tasks.center_out_reach.scalers import StandardVelocityScaler
from tasks.center_out_reach.settings import CenterOutReach
from tasks.center_out_reach.task_runner import TaskRunner
from tasks.center_out_reach.task_state import StateParams
from tasks.center_out_reach.task_state import TaskState
from tasks.center_out_reach.task_window import TaskWindow

from neural_data_simulator import inputs
from neural_data_simulator import outputs
from neural_data_simulator.outputs import StreamConfig
from neural_data_simulator.settings import LogLevel
from neural_data_simulator.util.runtime import configure_logger
from neural_data_simulator.util.runtime import initialize_logger
from neural_data_simulator.util.runtime import open_connection
from neural_data_simulator.util.runtime import unwrap
from neural_data_simulator.util.settings_loader import get_script_settings

SCRIPT_NAME = "nds-center-out-reach"
logger = logging.getLogger(__name__)


class _Settings(VersionedYamlModel):
    """Center-out reach app settings.

    Defines the schema of a `settings_center_out_reach.yaml` file.
    """

    log_level: LogLevel
    center_out_reach: CenterOutReach


def _setup_LSL_input(stream_name: str, connection_timeout: float) -> inputs.LSLInput:
    """Set up input that will read data from an LSL stream.

    Args:
        stream_name: The name of the LSL stream to read from.
        connection_timeout: The timeout in seconds to wait for
          the stream to become available.

    Returns:
        The input that can be used to read data from the LSL stream.
    """
    data_input = inputs.LSLInput(stream_name, connection_timeout)
    return data_input


def _setup_LSL_output(config: StreamConfig) -> outputs.LSLOutputDevice:
    """Set up output that will write data to an LSL stream.

    Args:
        config: The configuration for the LSL stream.

    Returns:
        The output that can be used to write to.
    """
    lsl_output = outputs.LSLOutputDevice(config)
    return lsl_output


def _get_task_window_params(
    task_settings: CenterOutReach.Task,
    window_settings: CenterOutReach.Window,
    unit_converter: PixelsToMetersConverter,
):
    return TaskWindow.Params(
        int(unit_converter.meters_to_pixels(task_settings.target_radius)),
        int(unit_converter.meters_to_pixels(task_settings.cursor_radius)),
        int(unit_converter.meters_to_pixels(task_settings.radius_to_target)),
        task_settings.number_of_targets,
        window_settings.colors.background,
        window_settings.colors.decoded_cursor,
        window_settings.colors.decoded_cursor_on_target,
        window_settings.colors.actual_cursor,
        window_settings.colors.target,
        window_settings.colors.target_waiting_for_cue,
        # if we use meter units for GUI elements they will be
        # correctly scaled to all devices
        font_size=int(unit_converter.meters_to_pixels(0.006)),
        button_size=(
            (
                int(unit_converter.meters_to_pixels(0.04)),
                int(unit_converter.meters_to_pixels(0.01)),
            )
        ),
        button_spacing=int(unit_converter.meters_to_pixels(0.005)),
        button_offset_top=int(unit_converter.meters_to_pixels(0.03)),
    )


def _get_state_machine_params(task_settings):
    return StateParams(
        task_settings.delay_to_begin,
        task_settings.delay_waiting_for_cue,
        task_settings.target_holding_time,
        task_settings.max_trial_time,
    )


def _get_window_rect(unit_converter, window_settings, task_settings) -> Tuple[int, int]:
    if window_settings.width and window_settings.height:
        return (
            window_settings.width,
            window_settings.height,
        )
    radius_to_target_pixels = unit_converter.meters_to_pixels(
        task_settings.radius_to_target
    )
    target_diameter_pixels = radius_to_target_pixels * 2
    return target_diameter_pixels * 1.2, target_diameter_pixels * 1.2


def _parse_rich_text(text: str, default_text_color: str):
    annotation_regex = re.compile(r"(\[[^\]]+\]\([^)]+\))")
    color_regex = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    text_parts = annotation_regex.split(text)
    hint = []
    for part in text_parts:
        if rp := list(color_regex.findall(part)):
            rich_text = rp[0]
            hint.append({"color": rich_text[1], "text": rich_text[0]})
        else:
            hint.append({"color": default_text_color, "text": part})
    return hint


def _get_menu_text(
    default_text_color,
    actual_cursor_color,
    decoded_cursor_color,
    target_color,
    input_device,
):
    text = (
        f"\nWelcome to the Center Out Reaching Task!\n\n"
        f"Press the <Start> button to begin.\n\n"
        f"Two cursors will be presented: the [input cursor]({actual_cursor_color})"
        f" that\n directly follows your {input_device} movements (and can be toggled\n"
        f" on and off by pressing `c` on the keyboard), and the\n"
        f" [decoded cursor]({decoded_cursor_color}) that is the decoded from the"
        f" simulated\n spikes from the [input cursor]({actual_cursor_color})"
        f" movement.\n\nYour goal is to reach and stay within the"
        f" [target]({target_color}) using the\n"
        f" [decoded cursor]({decoded_cursor_color})"
        f" until the next [target]({target_color}) is presented.\n"
        f"There is no time or target limit, you can press the\n"
        f" `Escape` key to finish the task at any time.\n"
    )
    return _parse_rich_text(text, default_text_color)


def _get_training_text(default_text_color, target_color, input_device):
    text = (
        f"\nWelcome to the Center Out Reaching Task!\n\n"
        f"Press the <Start> button to begin.\n\n"
        f"In open loop mode, the cursor follows your\n"
        f"{input_device} movements.\n\nYour goal is to reach and stay within the"
        f" [target]({target_color}) using the\n"
        f" cursor"
        f" until the next [target]({target_color}) is presented.\n\n"
        f"There is no time or target limit, you can press the\n"
        f" `Escape` key to finish the task at any time.\n"
    )
    return _parse_rich_text(text, default_text_color)


def _metrics_enabled(settings: _Settings) -> bool:
    return (
        settings.center_out_reach.input.enabled
        and settings.center_out_reach.with_metrics
    )


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run GUI.")
    parser.add_argument(
        "--settings-path",
        type=Path,
        help="Path to the settings_center_out_reach.yaml file.",
    )
    parser.add_argument(
        "--control-file",
        type=Path,
        help="Path to the control file that will receive control messages.",
    )
    args = parser.parse_args()
    return args


def run():
    """Run the center-out reach task GUI."""
    initialize_logger(SCRIPT_NAME)
    args = _parse_args()
    settings = cast(
        _Settings,
        get_script_settings(
            args.settings_path,
            "settings_center_out_reach.yaml",
            settings_parser=_Settings,
        ),
    )
    configure_logger(SCRIPT_NAME, settings.log_level)

    if settings.center_out_reach.input.enabled:
        lsl_input_settings = unwrap(settings.center_out_reach.input.lsl)
        data_input = _setup_LSL_input(
            lsl_input_settings.stream_name, lsl_input_settings.connection_timeout
        )
    else:
        data_input = None

    lsl_output_settings = settings.center_out_reach.output.lsl
    sampling_rate = settings.center_out_reach.sampling_rate
    data_output = _setup_LSL_output(
        StreamConfig.from_lsl_settings(
            lsl_output_settings,
            sampling_rate,
            n_channels=2,
        )
    )

    window_settings = settings.center_out_reach.window
    task_settings = settings.center_out_reach.task

    unit_converter = PixelsToMetersConverter(window_settings.ppi)
    window_rect = _get_window_rect(unit_converter, window_settings, task_settings)

    window_params = _get_task_window_params(
        task_settings,
        window_settings,
        unit_converter,
    )

    scaler_settings = settings.center_out_reach.standard_scaler
    velocity_scaler = StandardVelocityScaler(
        np.array(scaler_settings.scale), np.array(scaler_settings.mean), unit_converter
    )

    state_params = _get_state_machine_params(settings.center_out_reach.task)

    actual_cursor_color = window_settings.colors.actual_cursor
    decoded_cursor_color = window_settings.colors.decoded_cursor

    if _metrics_enabled(settings):
        metrics_collector = MetricsCollector(
            window_rect,
            task_settings.target_radius,
            unit_converter,
            actual_cursor_color,
            decoded_cursor_color,
        )
    else:
        metrics_collector = None

    with_decoded_cursor = settings.center_out_reach.input.enabled

    user_input = InputHandler()
    if with_decoded_cursor:
        menu_text = _get_menu_text(
            "black",
            actual_cursor_color,
            decoded_cursor_color,
            window_settings.colors.target,
            user_input.input_device_name,
        )
    else:
        menu_text = _get_training_text(
            "black", window_settings.colors.target, user_input.input_device_name
        )

    interrupted = False
    task_window = None
    try:
        with open_connection(data_output), open_connection(data_input):
            task_window = TaskWindow(window_rect, window_params, menu_text)
            task_state = TaskState(task_window, state_params)
            task_runner = TaskRunner(
                sampling_rate,
                data_input,
                data_output,
                velocity_scaler,
                with_decoded_cursor,
                metrics_collector,
            )
            logger.info("Running task")
            task_runner.run(task_state, user_input)
    except KeyboardInterrupt:
        logger.info("CTRL+C received. Exiting...")
        interrupted = True

    # This is used as a signal to a parent process that the main task has finished
    if args.control_file is not None:
        with open(args.control_file, "w") as control_file:
            control_file.write("main_task_finished\n")

    if (
        not interrupted
        and task_window is not None
        and not task_window.show_menu_screen
        and _metrics_enabled(settings)
    ):
        unwrap(metrics_collector).plot_metrics(task_window.target_positions)

    if task_window is not None:
        task_window.leave()


if __name__ == "__main__":
    run()
