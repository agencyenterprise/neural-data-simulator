"""Filter implementations for signal processing."""

from typing import Protocol, Tuple, Union

from numpy import ndarray
import numpy as np
from scipy import signal


class RealTimeFilter(Protocol):
    """A protocol for filters operating on chunked data.

    A python protocol (`PEP-544 <https://peps.python.org/pep-0544/>`_) works in
    a similar way to an abstract class.
    The :meth:`__init__` method of this protocol should never be called as
    protocols are not meant to be instantiated. An :meth:`__init__` method
    may be defined in a concrete implementation of this protocol if needed.
    """

    def execute(self, data: ndarray) -> ndarray:
        """Perform filtering on the current chunk of data.

        Args:
            data: Data that should be filtered.

        Returns:
            Filtered data with the same dimensions as the input.
        """
        ...


class GaussianFilter(RealTimeFilter):
    """An implementation of a Gaussian filter."""

    def __init__(
        self,
        *,
        name: str,
        window_size: int,
        std: float,
        normalization_coeff: float,
        num_channels: int,
        enabled: bool = True,
    ):
        """Initialize the GaussianFilter class.

        Args:
            name: A label that identifies the filter instance.
            window_size: The number of samples defining the size of the Gaussian window.
            std: Standard deviation.
            normalization_coeff: When applying the filter both numerator and
              denominator are normalized by this value.
            num_channels: Number of channels.
            enabled: Whether to apply the filter. If false, the
              data will be passed through without modification.
        """
        self.name = name
        self.window_size = window_size
        self.std = std
        self.normalization_coeff = normalization_coeff
        self.num_channels = num_channels
        self.enabled = enabled
        self._generate_filter_coefficients()

    def _generate_filter_coefficients(self):
        """Generate filter coefficients."""
        self.gauss_filter = signal.gaussian(self.window_size, self.std, sym=True)
        self._initialize_filter_state()

    def _initialize_filter_state(self):
        """Initialize filter state for each channel."""
        self._zi = signal.lfilter_zi(self.gauss_filter, 1)
        if self.num_channels > 1:
            self._zi = np.repeat(
                np.expand_dims(self._zi, axis=1), self.num_channels, axis=1
            )

    def execute(self, data: ndarray) -> ndarray:
        """Perform filtering on data.

        Args:
            data: Data that should be filtered as a two-dimensional array.
              The first dimension represents samples and the second
              dimension represents channels.

        Returns:
            Filtered data with the same dimensions as the input.
        """
        if not self.enabled:
            return data

        data, self._zi = signal.lfilter(
            self.gauss_filter, self.normalization_coeff, data, zi=self._zi, axis=0
        )

        return data


class ButterworthFilter(RealTimeFilter):
    """Generic class for Butterworth filters."""

    def __init__(
        self,
        *,
        name: str,
        filter_order: int,
        critical_frequency: Union[float, Tuple[float, float]],
        sample_rate: float,
        num_channels: int,
        btype: str,
        enabled: bool = True,
    ):
        """Perform Butterworth filtering.

        Args:
            name: A label that identifies the filter instance.
            filter_order: The order of the filter.
            critical_frequency: Critical frequency in Hz. For lowpass or highpass
              it is a scalar representing the cutoff frequency. For bandpass it is a
              tuple of two scalars representing the lower and upper cutoff frequencies.
            sample_rate: Sample rate in Hz.
            num_channels: Number of channels.
            btype: Type of filter. Either `highpass` or `lowpass`.
            enabled: Whether to apply the filter. If false, the
              data will be passed through without modification.
        """
        self.name = name
        self.filter_order = filter_order
        self.critical_frequency = critical_frequency
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.enabled = enabled
        self.btype = btype
        self._generate_filter_coefficients()

    def _generate_filter_coefficients(self):
        """Generate filter coefficents for filter using scipy."""
        self.filter_sos = signal.butter(
            self.filter_order,
            self.critical_frequency,
            btype=self.btype,
            analog=False,
            fs=self.sample_rate,
            output="sos",
        )
        self._initialize_filter_state()

    def _initialize_filter_state(self):
        """Initialize filter state for each channel."""
        self._zi = signal.sosfilt_zi(self.filter_sos)
        if self.num_channels > 1:
            self._zi = np.repeat(
                np.expand_dims(self._zi, axis=2), self.num_channels, axis=2
            )

    def execute(self, data: ndarray) -> ndarray:
        """Perform filtering on data.

        Args:
            data: Data that should be filtered as a two-dimensional array.
              The first dimension represents samples and the second
              dimension represents channels.

        Returns:
            Filtered data with the same dimensions as the input.
        """
        if not self.enabled:
            return data

        data, self._zi = signal.sosfilt(self.filter_sos, data, axis=0, zi=self._zi)

        return data


class HighpassFilter(ButterworthFilter):
    """Perform highpass filtering.

    This class is based around the scipy Butterworth digital
    filter implementation and uses the same language as the
    underlying scipy package.
    """

    def __init__(
        self,
        *,
        name: str,
        filter_order: int = 2,
        critical_frequency: float = 0.5,
        sample_rate: float,
        num_channels: int,
        enabled: bool = True,
    ):
        """Create a new instance.

        Args:
            name: A label that identifies the filter instance.
            filter_order: The order of the filter.
            critical_frequency: Critical frequency in Hz.
            sample_rate: Sample rate in Hz.
            num_channels: Number of channels.
            enabled: Whether to apply the filter. If false, the
              data will be passed through without modification.
        """
        super(HighpassFilter, self).__init__(
            name=name,
            filter_order=filter_order,
            critical_frequency=critical_frequency,
            sample_rate=sample_rate,
            num_channels=num_channels,
            enabled=enabled,
            btype="highpass",
        )


class LowpassFilter(ButterworthFilter):
    """Perform lowpass filtering.

    This class is based around the scipy Butterworth digital
    filter implementation and uses the same language as the
    underlying scipy package.
    """

    def __init__(
        self,
        *,
        name: str,
        filter_order: int = 2,
        critical_frequency: float = 50,
        sample_rate: float,
        num_channels: int,
        enabled: bool = True,
    ):
        """Perform lowpass filtering.

        Args:
            name: A label that identifies the filter instance.
            filter_order: The order of the filter.
            critical_frequency: Critical frequency in Hz.
            sample_rate: Sample rate in Hz.
            num_channels: Number of channels.
            enabled: Whether to apply the filter. If false, the
              data will be passed through without modification.
        """
        super(LowpassFilter, self).__init__(
            name=name,
            filter_order=filter_order,
            critical_frequency=critical_frequency,
            sample_rate=sample_rate,
            num_channels=num_channels,
            enabled=enabled,
            btype="lowpass",
        )


class BandpassFilter(ButterworthFilter):
    """Perform bandpass filtering.

    This class is based around the scipy Butterworth digital
    filter implementation and uses the same language as the
    underlying scipy package.
    """

    def __init__(
        self,
        *,
        name: str,
        filter_order: int = 2,
        critical_frequencies: Tuple[float, float],
        sample_rate: float,
        num_channels: int,
        enabled: bool = True,
    ):
        """Perform lowpass filtering.

        Args:
            name: A label that identifies the filter instance.
            filter_order: The order of the filter.
            critical_frequencies: Tuple of low and high critical frequencies in Hz.
            sample_rate: Sample rate in Hz.
            num_channels: Number of channels.
            enabled: Whether to apply the filter. If false, the
              data will be passed through without modification.
        """
        super(BandpassFilter, self).__init__(
            name=name,
            filter_order=filter_order,
            critical_frequency=critical_frequencies,
            sample_rate=sample_rate,
            num_channels=num_channels,
            enabled=enabled,
            btype="bandpass",
        )
