"""Stream data over LSL."""
from dataclasses import dataclass
import logging
from typing import List

import numpy as np
import pylsl

from nds.outputs import LSLOutputDevice
from nds.samples import Samples
from nds.timing import Timer

logger = logging.getLogger(__name__)


@dataclass
class Stream:
    """Stream information."""

    output: LSLOutputDevice
    """The output device to stream data to."""

    samples: Samples
    """The data with timestamps to stream."""

    is_finished: bool = False
    """Flag to indicate if the stream has finished."""

    read_cursor: int = 0
    """The current read cursor position."""

    @property
    def length(self):
        """The length of the data to stream.

        Returns:
            The number of data samples to stream.
        """
        return len(self.samples)


class LSLStreamer:
    """Streamer class that can be used to send samples through an LSL stream.

    A streamer class that takes a :class:`nds.samples.Samples` dataclass with timestamps
    and behavior data and stream it through LSL. Following the timestamps provided, the
    data is streamed to simulate a real-time data acquisition.
    """

    def __init__(
        self,
        outputs: List[LSLOutputDevice],
        samples: List[Samples],
        lsl_chunk_frequency: float,
        stream_indefinitely: bool,
    ) -> None:
        """Initialize an LSLStreamer class.

        Args:
            outputs: A list of LSL outputs to stream data out.
            samples: A list of Samples representing the data to stream.
            lsl_chunk_frequency: How often to send data to LSL outlets.
            stream_indefinitely: Set to True to continue streaming from the
              beginning after reaching the end of data.
        """
        if len(samples) != len(outputs):
            raise ValueError(
                "Number of Samples to stream needs to match number of outputs."
            )

        self.output_interval = 1.0 / lsl_chunk_frequency
        self.stream_indefinitely = stream_indefinitely

        self.streams: List[Stream] = []
        for i in range(len(outputs)):
            self.streams.append(Stream(outputs[i], samples[i]))

        self._timer = Timer(self.output_interval)

    def stream(self):
        """Stream samples.

        Stream all samples to all the outputs one output at a time.
        Irregular streams (sample rate is `0`) are streamed as `samples` to LSL
        while regular streams are streamed as `chunk`.
        """
        for stream in self.streams:
            logger.info(
                f"Streaming {stream.output.name}"
                f" with sample rate = {stream.output.sample_rate}",
            )

        self._timer.start()
        last_output_time = pylsl.local_clock()
        while not self._all_streams_finished:
            time_now = pylsl.local_clock()
            time_elapsed = time_now - last_output_time
            last_output_time = time_now
            self._stream_slice(time_elapsed, time_now)
            self._timer.wait()

    def _stream_slice(self, time_elapsed, time_now) -> None:
        for stream in self.streams:
            if stream.is_finished:
                continue

            if callable(stream.output.sample_rate):
                sample_rate = stream.output.sample_rate()
            else:
                sample_rate = stream.output.sample_rate

            if sample_rate > 0:
                self._stream_regular_slice(stream, time_elapsed, time_now)
            else:
                self._stream_irregular_slice(stream, time_elapsed)
            self._check_stream_finished(stream)

    def _stream_irregular_slice(self, stream: Stream, time_elapsed) -> None:
        slice_start_timestamp = stream.samples.timestamps[stream.read_cursor]
        stream_slice = np.logical_and(
            stream.samples.timestamps >= slice_start_timestamp,
            stream.samples.timestamps < (slice_start_timestamp + time_elapsed),
        )
        timestamps = stream.samples.timestamps[stream_slice]
        if len(timestamps) > 0:
            data = stream.samples.data[stream_slice]
            stream.read_cursor = stream.read_cursor + len(timestamps)
            stream.output.send(Samples(timestamps, data))

    def _stream_regular_slice(self, stream: Stream, time_elapsed, time_now) -> None:
        n_samples = np.rint(stream.output.sample_rate * time_elapsed).astype(int)
        if n_samples > 0:
            new_cursor = min(stream.read_cursor + n_samples, stream.length)
            stream_slice = np.arange(start=stream.read_cursor, stop=new_cursor)
            stream.read_cursor = new_cursor
            if len(stream_slice) > 0:
                timestamps = stream.samples.timestamps[stream_slice]
                timestamps = timestamps - timestamps[-1] + time_now
                data = stream.samples.data[stream_slice]
                stream.output.send_as_chunk(data, timestamps[0])

    def _check_stream_finished(self, stream: Stream):
        if stream.read_cursor >= stream.length:
            logger.info(f"{stream.output.name} finished")
            if self.stream_indefinitely:
                logger.info("Restarting streaming from the beginning")
                stream.read_cursor = 0
            else:
                stream.is_finished = True

    @property
    def _all_streams_finished(self):
        for stream in self.streams:
            if not stream.is_finished:
                return False
        return True
