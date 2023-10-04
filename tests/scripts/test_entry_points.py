"""Test that all entry points defined in pyproject.toml exist.""" ""
import importlib
import pathlib

import pytest
import toml


@pytest.fixture
def pyproject_toml_path() -> pathlib.Path:
    """Return the path to this repository's pyproject.toml file."""
    file_dir = pathlib.Path(__file__).parent
    project_root = file_dir.parent.parent
    return project_root.joinpath("pyproject.toml")


def test_check_entry_points(pyproject_toml_path: pathlib.Path):
    """Test that all entry points defined in pyproject.toml exist."""
    with pyproject_toml_path.open("r") as toml_file:
        config = toml.load(toml_file)

    # Get the script entry points from the loaded TOML file
    scripts = config.get("tool", {}).get("poetry", {}).get("scripts", {})
    assert scripts, "No scripts/entry points defined in pyproject.toml"

    for entry_point_name, script_path in scripts.items():
        # Split the script path into module and function parts
        # https://python-poetry.org/docs/pyproject/#scripts
        module_name, function_name = script_path.split(":")

        try:
            # Load the module dynamically using importlib
            module = importlib.import_module(module_name)

            # Check if the function exists in the module
            assert hasattr(module, function_name), (
                f"Entry-point function '{function_name}'"
                f"not found in module '{module_name}'."
            )
        except ImportError as exc:
            raise ImportError(f"Entry point '{entry_point_name}' does not exist: {exc}")
