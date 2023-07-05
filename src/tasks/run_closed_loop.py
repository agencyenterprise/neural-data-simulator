"""Run all components of the BCI closed loop."""
import subprocess
import sys


def _run_process(args) -> subprocess.Popen:
    return subprocess.Popen(args, shell=sys.platform == "win32")


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

    encoder = _run_process(["encoder"])
    ephys = _run_process(["ephys_generator"])
    decoder = _run_process(["decoder"])
    center_out_reach = _run_process(["center_out_reach"])

    center_out_reach.wait()
    encoder.kill()
    ephys.kill()
    decoder.kill()


if __name__ == "__main__":
    run()
