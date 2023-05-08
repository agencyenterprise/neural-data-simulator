"""Parse settings from files."""
import errno
import logging
from pathlib import Path
from typing import Optional, Type

from pydantic_yaml import VersionedYamlModel

from neural_data_simulator.errors import UnexpectedSettingsVersion
from neural_data_simulator.util.runtime import get_abs_path
from neural_data_simulator.util.runtime import NDS_HOME

logger = logging.getLogger(__name__)


def get_script_settings(
    settings_file: Optional[Path],
    filename: str,
    settings_parser: Type[VersionedYamlModel],
):
    """Load script settings from the filepath."""
    if settings_file is None:
        default_settings_file = Path(f"{NDS_HOME}/{filename}")
        if not default_settings_file.exists():
            raise FileNotFoundError(
                errno.ENOENT,
                "Default settings file not found. Run 'nds_post_install_config' "
                "to copy the default settings files.\n"
                "Alternatively, you can specify the path to the settings file via the "
                "'--settings-path' argument.",
                str(default_settings_file),
            )

        settings_file = default_settings_file

    with open(get_abs_path(str(settings_file))) as f:
        try:
            settings = settings_parser.parse_raw(f.read(), proto="yaml")
        except UnexpectedSettingsVersion as e:
            raise ValueError(
                f"The settings version {e.loaded_version} doesn't match "
                f"the expected version {e.expected_version}.\nPlease migrate "
                f"the file {settings_file} to the new version or reset it "
                f"by running 'nds_post_install_config --overwrite-existing-files'.\n"
                f"For information on how to migrate the settings file, see the "
                f"release notes."
            )

    logger.info(f"Using settings from '{settings_file}'")

    return settings
