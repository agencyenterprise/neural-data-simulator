"""Script for saving data with timestamps from LSL streams into a file."""
import argparse
import signal
import sys
import time

import numpy as np

from neural_data_simulator import inputs


class LSLStreamRecorder:
    """Helper class for collecting data from an LSL stream."""

    def __init__(self, stream_name):
        """Initialize the LSLStreamRecorder class.

        Args:
            stream_name: Name of the LSL stream to record.
        """
        lsl_input = inputs.LSLInput(stream_name, 60.0)
        lsl_input.connect()
        self.input = lsl_input
        self.data = np.array([]).reshape(0, lsl_input.get_info().channel_count)
        self.stream_name = stream_name
        self.timestamps = np.array([])

    def collect_sample(self):
        """Try to read and store samples from the LSL stream."""
        data_samples = self.input.read()
        if not data_samples.empty:
            self.data = np.vstack((self.data, data_samples.data))
            self.timestamps = np.concatenate((self.timestamps, data_samples.timestamps))

    def save(self, prefix=""):
        """Save the collected data to an `npz` file.

        The file will be named `prefix` + `stream_name` + `.npz`.

        Args:
            prefix: Prefix to add to the filename.
        """
        np.savez(
            f"{prefix}{self.stream_name}.npz",
            timestamps=np.array(self.timestamps, dtype=float),
            data=self.data,
        )


def _persist_session(recorders, session_name):
    for recorder in recorders:
        recorder.save(f"{session_name}_")


def _handle_sigterm(recorders, session_name):
    def sigterm_handler(_signo, _stack_frame):
        _persist_session(recorders, session_name)
        sys.exit(0)

    return sigterm_handler


def run():
    """Start the recorder."""
    parser = argparse.ArgumentParser(description="Run recorder.")
    parser.add_argument(
        "--recording-time",
        type=int,
        default=1000,
        help="Number of seconds to record.",
    )
    parser.add_argument(
        "--session",
        type=str,
        required=True,
        help="Name of the recording session, used for the output filename.",
    )
    parser.add_argument(
        "--lsl",
        type=str,
        required=True,
        help="Name of the LSL stream(s) to record, use ',' for a list.",
    )

    parsed_args = parser.parse_args()
    recording_time = parsed_args.recording_time
    session_name = parsed_args.session
    lsl_streams = parsed_args.lsl.split(",")

    recorders = [LSLStreamRecorder(stream_name.strip()) for stream_name in lsl_streams]

    signal.signal(signal.SIGTERM, _handle_sigterm(recorders, session_name))

    start_time = time.perf_counter()
    print("will start to record now")
    try:
        while True:
            for recorder in recorders:
                recorder.collect_sample()

            if time.perf_counter() - start_time >= recording_time:
                break
    except KeyboardInterrupt:
        print("saving data now")

    print(f"Samples / s: {len(recorders[0].data) / (time.perf_counter() - start_time)}")
    _persist_session(recorders, session_name)


if __name__ == "__main__":
    run()
