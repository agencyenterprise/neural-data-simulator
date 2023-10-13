"""Script for saving data with timestamps from LSL streams into a file."""
import argparse
import signal
import sys
import time

from neural_data_simulator.recorder.recorders import LSLStreamRecorder


def _persist_session(recorders, session_name):
    for recorder in recorders:
        recorder.save(f"{session_name}_")


def _handle_sigterm(recorders, session_name):
    def sigterm_handler(_signo, _stack_frame):
        _persist_session(recorders, session_name)
        sys.exit(0)

    return sigterm_handler


def _parse_args():
    """Parse command line arguments."""
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
        nargs="+",
        help="Name(s) of the LSL stream(s) to record, separate with spacesfor a list.",
    )

    return parser.parse_args()


def run():
    """Start the recorder."""
    args = _parse_args()
    recording_time = args.recording_time
    session_name = args.session
    lsl_streams = args.lsl

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
