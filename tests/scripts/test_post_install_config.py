"""Test for post_install_config.py."""

import argparse
import os
from pathlib import Path

import pytest

from nds.scripts import post_install_config
from nds.util import runtime


@pytest.fixture(autouse=True)
def fake_parse_args(monkeypatch: pytest.MonkeyPatch) -> argparse.Namespace:
    """Fake command line arguments passed to the script."""
    parse_args_result = argparse.Namespace(
        overwrite_existing_files=False,
        ignore_extras_config=False,
        ignore_sample_data_download=False,
    )

    def parse_args(self, args=None, namespace=None):
        return parse_args_result

    monkeypatch.setattr("argparse.ArgumentParser.parse_args", parse_args)
    return parse_args_result


@pytest.fixture(autouse=True)
def tmp_nds_home(tmp_path):
    """Set the NDS_HOME constant to a temporary path."""
    tmp_nds_home = os.path.join(tmp_path, "nds")
    runtime.NDS_HOME = tmp_nds_home
    return tmp_nds_home


@pytest.fixture(autouse=True)
def fake_pooch_retrieve(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Fake the pooch retrieve implementation in the script."""

    def retrieve(url: str, known_hash: str, fname: str) -> str:
        file_path = os.path.join(tmp_path, fname)
        Path(file_path).touch()
        return file_path

    monkeypatch.setattr("nds.scripts.post_install_config.pooch.retrieve", retrieve)


@pytest.fixture()
def config_files():
    """Return a list of expected config files."""
    return ["settings.yaml", "settings_streamer.yaml", "settings_decoder.yaml"]


@pytest.fixture()
def sample_data_files():
    """Return a list of expected sample data files."""
    return [
        "session_4_behavior_standardized.npz",
        "session_4_tuning_curves_params.npz",
        "session_4_simple_decoder.joblib",
    ]


@pytest.fixture()
def expected_config_paths(tmp_nds_home, config_files):
    """Return a list of expected config file paths."""
    return [os.path.join(tmp_nds_home, config_file) for config_file in config_files]


@pytest.fixture()
def expected_sample_data_paths(tmp_nds_home, sample_data_files):
    """Return a list of expected config file paths."""
    return [
        os.path.join(tmp_nds_home, "sample_data", sample_data)
        for sample_data in sample_data_files
    ]


def get_file_attributes(file_paths):
    """Return a list of file attributes for the given file paths."""
    return [
        {
            "size": os.path.getsize(file_path),
            "mtime": os.path.getmtime(file_path),
            "ctime": os.path.getctime(file_path),
            "st_atime_ns": os.stat(file_path).st_atime_ns,
        }
        for file_path in file_paths
    ]


class TestPostInstallConfig:
    """Test running the post_install_config script."""

    def test_run_first_install(
        self, tmp_nds_home, expected_config_paths, expected_sample_data_paths
    ):
        """Test run when the NDS_HOME folder does not exist."""
        post_install_config.run()

        assert os.path.exists(tmp_nds_home)
        file_paths = expected_config_paths + expected_sample_data_paths

        for file_path in file_paths:
            assert os.path.exists(file_path)

        # change copied files
        for file_path in file_paths:
            with open(file_path, "w") as f:
                f.write("c")
        attributes = get_file_attributes(file_paths)

        # running post_install_config a second time shouldn't change the file attributes
        post_install_config.run()

        new_attributes = get_file_attributes(file_paths)
        assert attributes == new_attributes

    def test_run_overwrite_existing_files(
        self, fake_parse_args, expected_config_paths, expected_sample_data_paths
    ):
        """Test run with --overwrite_existing_files when all required files exist."""
        post_install_config.run()

        file_paths = expected_config_paths + expected_sample_data_paths

        # change copied files
        for file_path in file_paths:
            with open(file_path, "w") as f:
                f.write("c")
        attributes = get_file_attributes(file_paths)

        # running post_install_config a second time should overwrite the file attributes
        fake_parse_args.overwrite_existing_files = True
        post_install_config.run()

        new_attributes = get_file_attributes(file_paths)
        assert attributes != new_attributes

    def test_run_ignore_extras_config(
        self, expected_config_paths, expected_sample_data_paths, fake_parse_args
    ):
        """Test run with --ignore_extras_config."""
        fake_parse_args.ignore_extras_config = True
        post_install_config.run()

        file_paths = expected_config_paths + expected_sample_data_paths

        for file_path in file_paths:
            if file_path.endswith("settings_decoder.yaml"):
                assert not os.path.exists(file_path)
            else:
                assert os.path.exists(file_path)

    def test_run_ignore_sample_data_download(
        self, expected_config_paths, expected_sample_data_paths, fake_parse_args
    ):
        """Test run with --ignore_sample_data_download."""
        fake_parse_args.ignore_sample_data_download = True
        post_install_config.run()

        for file_path in expected_config_paths:
            assert os.path.exists(file_path)

        for file_path in expected_sample_data_paths:
            assert not os.path.exists(file_path)
