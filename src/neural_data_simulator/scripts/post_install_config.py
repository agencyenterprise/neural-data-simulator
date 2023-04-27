"""Script to bootstrap the NDS environment.

This script needs to be run by the user after installing NDS.
It will create the NDS_HOME directory in `$HOME/.nds` then copy the default
configuration files to it. In addition, it will download the example models
and behavior data, validate their `md5` hashes, and copy them to the
`NDS_HOME/sample_data` directory.
"""


import argparse
import os
import shutil
from urllib.parse import urljoin

import decoder
import plugins
import pooch
import tasks

import neural_data_simulator
from neural_data_simulator.util.runtime import get_abs_path
from neural_data_simulator.util.runtime import get_configs_dir
from neural_data_simulator.util.runtime import get_plugins_dir
from neural_data_simulator.util.runtime import get_sample_data_dir

plugin_files = [
    ("model.py", plugins.__file__),
    ("preprocessor.py", plugins.__file__),
    ("postprocessor.py", plugins.__file__),
    ("gamepad_preprocessor.py", plugins.__file__),
]

plugin_test_files = [
    ("test_model.py", os.path.join(os.path.dirname(plugins.__file__), "examples"))
]


core_configs = [
    ("settings.yaml", neural_data_simulator.__file__),
    ("settings_streamer.yaml", neural_data_simulator.__file__),
    ("lsl.config", neural_data_simulator.__file__),
]

extras_configs = [
    ("settings_decoder.yaml", decoder.__file__),
    ("settings_center_out_reach.yaml", tasks.__file__),
]

download_base_url = "https://neural-data-simulator.s3.amazonaws.com/sample_data/v1/"

sample_data = {
    "session_4_behavior_standardized.npz": "md5:2f5c5eb913e55fe9ec2336ea743d72ce",
    "session_4_tuning_curves_params.npz": "md5:93b671e3fba89b6114bd9cfb17770876",
    "session_4_simple_decoder.joblib": "md5:738d624dac89c9164f1dbca3104cdb83",
}


def _download_sample_data(overwrite: bool):
    sample_data_dir = get_sample_data_dir()
    os.makedirs(sample_data_dir, exist_ok=True)
    for filename, hash_ in sample_data.items():
        local_file_path = os.path.join(sample_data_dir, filename)
        if not os.path.exists(local_file_path) or overwrite:
            url = urljoin(download_base_url, filename)
            downloaded_file_path = pooch.retrieve(
                url=url, known_hash=hash_, fname=filename
            )
            shutil.copy(
                downloaded_file_path,
                sample_data_dir,
            )
            print(f"Copied '{filename}' to {local_file_path}")
        else:
            print(
                f"Skipped '{filename}' because it already exists in {local_file_path}"
            )


def _copy_files(
    file_list: list, parent_dir: str, destination_dir: str, overwrite: bool
):
    os.makedirs(destination_dir, exist_ok=True)
    for file_name, file_parent in file_list:
        config_file_path = os.path.join(destination_dir, file_name)
        if not os.path.exists(config_file_path) or overwrite:
            shutil.copy(
                get_abs_path(os.path.join(parent_dir, file_name), file_parent),
                destination_dir,
            )
            print(f"Copied '{file_name}' to {destination_dir}")
        else:
            print(
                f"Skipped '{file_name}' because it"
                f" already exists in {destination_dir}"
            )


def run():
    """Copy config files and sample data to NDS_HOME."""
    parser = argparse.ArgumentParser(description="Run post install steps.")
    parser.add_argument(
        "--ignore-extras-config",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ignore config files for decoder and center-out-reach GUI.",
    )
    parser.add_argument(
        "--ignore-sample-data-download",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Don't download sample data for the BCI closed loop.",
    )
    parser.add_argument(
        "--overwrite-existing-files",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace existing config files and sample data if they exist.",
    )
    args = parser.parse_args()

    _copy_files(
        core_configs, "config", get_configs_dir(), args.overwrite_existing_files
    )
    if not args.ignore_extras_config:
        _copy_files(
            extras_configs, "config", get_configs_dir(), args.overwrite_existing_files
        )

    _copy_files(
        plugin_files, "examples", get_plugins_dir(), args.overwrite_existing_files
    )
    _copy_files(
        plugin_test_files,
        "tests",
        os.path.join(get_plugins_dir(), "tests"),
        args.overwrite_existing_files,
    )

    if not args.ignore_sample_data_download:
        _download_sample_data(args.overwrite_existing_files)


if __name__ == "__main__":
    run()
