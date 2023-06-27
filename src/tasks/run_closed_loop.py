"""Run all components of the BCI closed loop."""
import subprocess
import sys


def start_process(command):
    """Start a new process passing this current process arguments."""
    return subprocess.Popen([command] + sys.argv[1:])


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

    encoder = start_process("encoder")
    ephys = start_process("ephys_generator")
    decoder = start_process("decoder")
    center_out_reach = start_process("center_out_reach")

    center_out_reach.wait()
    encoder.kill()
    ephys.kill()
    decoder.kill()


if __name__ == "__main__":
    run()
