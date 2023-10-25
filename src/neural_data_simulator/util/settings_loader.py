"""Parse settings from files."""
import argparse
import logging
import os
from typing import Optional, Type

from omegaconf import OmegaConf
from pydantic import BaseModel
import yaml

from neural_data_simulator.util.runtime import get_configs_dir

logger = logging.getLogger(__name__)


def load_settings(
    settings_file: os.PathLike,
    override_dotlist: Optional[list[str]],
    settings_parser: Type[BaseModel],
):
    """Load script settings with optional overrides."""
    try:
        with open(settings_file, "r") as f:
            settings_dict = yaml.safe_load(f)
    except FileNotFoundError as file_error:
        raise FileNotFoundError(
            f"Settings file not found: {settings_file}.\n"
            "\tRun 'nds_post_install_config' to copy the default settings files\n"
            f"\tto: {get_configs_dir()}\n"
            "\tAlternatively, you can specify the path to the settings file via the\n"
            "\t'--settings-path' argument."
        ) from file_error

    settings = settings_parser.parse_obj(settings_dict)

    if override_dotlist:
        override_conf = OmegaConf.from_dotlist(override_dotlist)
        merged_conf = OmegaConf.merge(settings.dict(), override_conf)
        # Re-validate merged config
        # If validation is slow, we can use `pydantic-partial` to only
        # re-validate the dot-list overrides.
        settings = settings_parser.parse_obj(merged_conf)

    return settings


def check_config_override_str(value: str) -> str:
    """Custom argparse type to check for individual dot-list arguments."""
    parts = value.split("=")
    try:
        key, val = parts
    except ValueError:
        # not enough / too many values to unpack
        raise argparse.ArgumentTypeError(
            f"Invalid config-override: {value}\n"
            "\tExpected format: `key=value` or `key.subkey=value`"
        )

    key_parts = key.split(".")
    if not key_parts or "" in key_parts:
        raise argparse.ArgumentTypeError(
            f"Invalid config-override: {value}\n"
            "\tExpected format: `key=value` or `key.subkey=value`"
        )
    return value
