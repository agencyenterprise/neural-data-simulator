"""Classes for recording data from source(s)."""

import numpy as np

from neural_data_simulator.core.inputs.lsl_input import LSLInput


class LSLStreamRecorder:
    """Helper class for collecting data from an LSL stream."""

    def __init__(self, stream_name):
        """Initialize the LSLStreamRecorder class.

        Args:
            stream_name: Name of the LSL stream to record.
        """
        lsl_input = LSLInput(stream_name, 60.0)
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
