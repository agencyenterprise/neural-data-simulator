"""Migrations for the settings files."""
from neural_data_simulator.config.migrations import _v100_to_110

MIGRATIONS = {
    "settings.yaml": {
        "1.0.0": _v100_to_110,
        # add new migrations here
    }
}
