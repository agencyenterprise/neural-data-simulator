"""Run all components of the BCI closed loop."""
import subprocess


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

    encoder = subprocess.Popen(["encoder"])
    ephys = subprocess.Popen(["ephys_generator"])
    decoder = subprocess.Popen(["decoder"])
    center_out_reach = subprocess.Popen(["center_out_reach"])

    center_out_reach.wait()
    encoder.kill()
    ephys.kill()
    decoder.kill()


if __name__ == "__main__":
    run()
