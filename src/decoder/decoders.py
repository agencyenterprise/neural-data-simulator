"""This module contains the included decoder implementation -- the :class:`Decoder`."""
import logging
import time
from typing import Protocol

import joblib
from numpy import ndarray
import numpy as np

from neural_data_simulator.filters import BandpassFilter
from neural_data_simulator.filters import GaussianFilter
from neural_data_simulator.filters import LowpassFilter
from neural_data_simulator.samples import Samples
from neural_data_simulator.util.buffer import RingBuffer

logger = logging.getLogger(__name__)


class DecoderModel(Protocol):
    """Protocol for a `Decoder` model.

    A `Decoder model` predicts behavior data from spike rate data.

    The Decoder processes data in chunks represented as
    :class:`neural_data_simulator.samples.Samples`.
    One chunk may contain several spike rate data points (n_samples) across multiple
    units (n_units). The :meth:`predict` method is called for each chunk in order to
    transform the spike rate data into behavior data (n_samples) across multiple axes
    (n_axes).

    A python protocol (`PEP-544 <https://peps.python.org/pep-0544/>`_) works in
    a similar way to an abstract class.
    The :meth:`__init__` method of this protocol should never be called as
    protocols are not meant to be instantiated. An :meth:`__init__` method
    may be defined in a concrete implementation of this protocol if needed.
    """

    def predict(self, data: ndarray) -> ndarray:
        """Predict behavior from spike rate input.

        Args:
            data: Spike rate data as :class:`neural_data_simulator.samples.Samples`
                with shape (n_samples, n_units).

        Returns:
            Behavior data as :class:`neural_data_simulator.samples.Samples` with shape
            (n_samples, n_axes). For example, in case of modeling velocities in a
            horizontal and vertical direction (2 axes), the returned data is a 2D
            array with shape (n_samples, 2).
        """
        ...


class PersistedFileDecoderModel(DecoderModel):
    """Decoder pre-trained model that can be loaded from a file."""

    def __init__(self, model_file_path: str) -> None:
        """Initialize the model from a file path.

        Args:
            model_file_path: The path to the model file.
        """
        super().__init__()
        self.model: DecoderModel = joblib.load(model_file_path)

    def predict(self, data: ndarray) -> ndarray:
        """Predict behavior from spike rate input.

        Args:
            data: A spike rate vector.

        Returns:
            A behavior vector.
        """
        return self.model.predict(data)[0]


class Decoder:
    """Decoder implementation that decodes spiking raw data into velocities."""

    def __init__(
        self,
        model: DecoderModel,
        input_sample_rate: float,
        output_sample_rate: float,
        n_channels: int,
        threshold: float,
    ):
        """Initialize the decoder.

        Args:
            model: The decoder model.
            input_sample_rate: The input sample rate in Hz.
            output_sample_rate: The output sample rate in Hz.
            n_channels: The number of input channels.
            threshold: The spike detection threshold.
        """
        self.model = model
        self.output_sample_rate = output_sample_rate
        self.threshold = threshold
        self.bin_width = 0.02
        self.n_channels = n_channels
        self.decoding_window_size = int(input_sample_rate / output_sample_rate)

        self.rates_filter = GaussianFilter(
            name="gauss_filter",
            window_size=6,
            std=3,
            normalization_coeff=6,
            num_channels=n_channels,
            enabled=True,
        )

        self.velocities_filter = LowpassFilter(
            name="lp_filter",
            filter_order=6,
            critical_frequency=5,
            sample_rate=output_sample_rate,
            num_channels=2,
            enabled=True,
        )

        self.raw_data_filter = BandpassFilter(
            name="bp_filter",
            filter_order=1,
            critical_frequencies=(250, 2000),
            sample_rate=input_sample_rate,
            num_channels=n_channels,
            enabled=True,
        )

        self._buffer = RingBuffer(
            max_samples=int(
                self.decoding_window_size * 100
            ),  # 2 seconds of data in the decoder buffer
            n_channels=n_channels + 1,
        )

        self.last_run_time = time.perf_counter_ns()

    def _add_to_buffer(self, samples: Samples):
        if not samples.empty:
            self._buffer.add(np.column_stack((samples.timestamps, samples.data)))

    def _consume_from_buffer(self) -> Samples:
        if len(self._buffer) < self.decoding_window_size:
            return Samples.empty_samples()
        timestamps_and_data = self._buffer.read(self.decoding_window_size)
        timestamps = timestamps_and_data[:, 0]
        data = timestamps_and_data[:, 1:]
        return Samples(timestamps, data)

    def decode(self, samples: Samples) -> Samples:
        """Decode spiking raw data into velocities.

        Args:
            samples: The raw data samples to decode.

        Returns:
            The decoded velocities.
        """
        time_now = time.perf_counter_ns()
        elapsed_time_ns = time_now - self.last_run_time

        self._add_to_buffer(samples)
        logger.debug(f"elapsed_time_ns={elapsed_time_ns} buffer={len(self._buffer)}")

        decoded_velocities = []
        decoded_timestamps = []
        while True:
            samples = self._consume_from_buffer()
            if samples.empty:
                break

            timestamps = samples.timestamps
            data = samples.data
            bin_rates = np.zeros((1, self.n_channels))
            duration = timestamps[-1] - timestamps[0]

            if duration > 0:
                filtered_data = self.raw_data_filter.execute(data)
                bin_rates = self._get_bin_rates(filtered_data, duration)

            bin_rates = self.rates_filter.execute(bin_rates)

            velocities = self.model.predict(bin_rates).reshape(1, 2)
            velocities = self.velocities_filter.execute(velocities)
            decoded_velocities.append(velocities[0])
            decoded_timestamps.append(timestamps[-1])

        self.last_run_time = time_now
        return Samples(
            timestamps=np.array(decoded_timestamps), data=np.array(decoded_velocities)
        )

    def _get_bin_rates(
        self,
        samples: ndarray,
        duration: float,
    ) -> ndarray:
        bin_rates = []

        for channel in range(self.n_channels):
            spike_indices = self._threshold_crossing(samples[:, channel])
            rate = len(spike_indices) / duration
            bin_rates.append(rate)
        return np.array(bin_rates).reshape(1, self.n_channels)

    def _threshold_crossing(self, a: ndarray) -> ndarray:
        """Compute the indices of the array where the values pass a threshold.

        Args:
            a: An one-dimensional array.

        Returns:
            The array indices that correspond to a crossing.

        Raises:
            ValueError: If the threshold is zero.
        """
        if self.threshold > 0:
            return (
                np.nonzero((a[1:] >= self.threshold) & (a[:-1] < self.threshold))[0] + 1
            )
        elif self.threshold < 0:
            return (
                np.nonzero((a[1:] <= self.threshold) & (a[:-1] > self.threshold))[0] + 1
            )
        else:
            raise ValueError("Threshold must be non-zero")
