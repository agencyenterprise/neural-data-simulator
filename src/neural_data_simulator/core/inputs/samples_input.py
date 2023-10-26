"""Input implementation for :class:`neural_data_simulator.samples.Samples`."""

import logging
import time
from typing import Optional

import numpy as np

from neural_data_simulator.core.inputs.api import Input
from neural_data_simulator.core.samples import Samples


class SamplesInput(Input):
    """An input object based on a :class:`neural_data_simulator.samples.Samples`.

    The underlying samples dataclass will have its timestamps modified to be
    in reference to when the first read was made from this class, simulating the
    appearance of data being collected in real-time. Alternatively, the function
    `set_reference_time_to_now` can be called prior to the first `read` of the data to
    use that as a reference time.

    A timer is synced between the reference time and the first timestamp in the input
    samples. Any calls to the `read` function will calculate the current time in
    reference to the synced timer and return the appropriate samples.
    """

    def __init__(self, input_samples: Samples) -> None:
        """Initialize the SamplesInput class.

        Args:
            input_samples: Dataclass containing timestamps and behavior data.
        """
        self._input_samples = input_samples
        self._logger = logging.getLogger(__name__)
        self._index_next_sample_to_read = 0
        self._has_reference_time = False

        self._last_time_streamed: Optional[float] = None
        self._time_reference: Optional[float] = None

    def set_reference_time_to_now(self):
        """Set current time as starting time for data stream."""
        self._initialize_streaming_time()
        self._set_reference_time()
        self._has_reference_time = True

    def read(self) -> Samples:
        """Get new samples from the time of last read.

        If first call to `read` samples will be read since the call to
        `set_reference_time_to_now`.
        If `set_reference_time_to_now` was not previously called, it will be called.

        Returns:
            :class:`neural_data_simulator.samples.Samples` dataclass with timestamps and
            data available since last `read` call.
        """
        if not self._has_reference_time:
            self._logger.warning(
                "Reference time not calculated before first read call."
                " Calculating now."
            )
            self.set_reference_time_to_now()

        index_current_sample = self._get_current_sample_index()
        output_samples = self._get_samples_from_last_read_to_index(index_current_sample)
        self._index_next_sample_to_read = index_current_sample + 1

        return output_samples

    def _get_current_sample_index(self) -> int:
        """Find sample index corresponding to current time.

        Get index of sample representing current time relative to input samples
        time.
        """
        time_now = self._get_referenced_time()
        return int(
            np.searchsorted(self._input_samples.timestamps, time_now, side="right") - 1
        )

    def _get_samples_from_last_read_to_index(self, index: int) -> Samples:
        timestamps = self._input_samples.timestamps[
            self._index_next_sample_to_read : index + 1
        ]
        data = self._input_samples.data[self._index_next_sample_to_read : index + 1]

        return Samples(timestamps, data)

    def _initialize_streaming_time(self) -> None:
        self._last_time_streamed = time.time()

    def _set_reference_time(self) -> None:
        """Set reference time.

        Reference time is set as the time difference from `set_reference_time_to_now`
        call to first timestamp in the input samples.
        This correction is used to convert real time elapsed to time elapsed relative to
        the samples first timestamp.
        """
        if self._last_time_streamed:
            self._time_reference = (
                self._last_time_streamed - self._input_samples.timestamps[0]
            )
        else:
            raise ValueError("Streaming time not initialized")

    def _get_referenced_time(self) -> float:
        """Get current time in the input samples reference."""
        if self._time_reference:
            return time.time() - self._time_reference
        else:
            raise ValueError("Reference time not initialized")

    def connect(self) -> None:
        """No action required during connect for this class."""
        pass
