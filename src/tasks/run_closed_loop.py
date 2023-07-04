"""Run all components of the BCI closed loop."""
import argparse
from pathlib import Path
import subprocess


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
        help="Path to the settings.yaml file with the task config.",
    )
    return parser.parse_args()


def _build_param_from_arg(args, arg_property, param_name):
    params = []
    attr_value = getattr(args, arg_property, None)
    if attr_value is not None:
        params = [param_name, str(attr_value)]
    return params


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

    args = _parse_args()

    SETTINGS_PATH_PARAM = "--settings-path"

    nds_params = _build_param_from_arg(args, "nds_settings_path", SETTINGS_PATH_PARAM)
    decoder_params = _build_param_from_arg(
        args, "decoder_settings_path", SETTINGS_PATH_PARAM
    )
    task_params = _build_param_from_arg(args, "task_settings_path", SETTINGS_PATH_PARAM)

    encoder = subprocess.Popen(["encoder"] + nds_params)
    ephys = subprocess.Popen(["ephys_generator"] + nds_params)
    decoder = subprocess.Popen(["decoder"] + decoder_params)
    center_out_reach = subprocess.Popen(["center_out_reach"] + task_params)

    center_out_reach.wait()
    encoder.kill()
    ephys.kill()
    decoder.kill()


if __name__ == "__main__":
    run()
