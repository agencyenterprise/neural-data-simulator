"""This module contains classes that can be used to generate noise."""
from typing import Optional, Protocol, Tuple

import colorednoise
from numpy import ndarray
import numpy as np
from scipy import signal
import scipy.fft
import scipy.signal


def synthesize_signal(psd_one_sided: ndarray, n_samples: int) -> ndarray:
    """Synthesize a signal from a given PSD.

    Args:
        psd_one_sided: The one-sided power spectral density of the signal.
        n_samples: The number of samples to generate.

    Returns:
        The signal in the time domain.
    """
    n_samples_psd_one_sided = len(psd_one_sided)
    psd = _get_two_sided_psd(psd_one_sided)
    psd = signal.resample(psd, n_samples)
    freq_spectrum = np.sqrt(psd)
    return _freq_to_time(freq_spectrum, n_samples_psd_one_sided)


def calc_psd(y: ndarray, fs: float) -> Tuple[ndarray, ndarray]:
    """Calculate the power spectral density of a signal.

    Args:
        y: The signal.
        fs: The sampling frequency.

    Returns:
        The frequencies and the power spectral density.
    """
    freq_spectrum = scipy.fft.rfft(y) * 1 / len(y)
    PSD = np.abs(freq_spectrum) ** 2
    frequencies = scipy.fft.rfftfreq(len(y), 1 / fs)
    return (frequencies, PSD)


def _get_two_sided_psd(psd_one_sided: ndarray) -> ndarray:
    psd_one_sided = psd_one_sided.astype(complex)
    # Multiply the values corresponding to the zero and Nyquist frequencies
    # by 2 to compensate for the fact that they are only present once in the PSD
    psd_one_sided[0] *= 2
    psd_one_sided[-1] *= 2
    # Mirror the one sided PSD to create the negative frequencies
    psd = np.concatenate((psd_one_sided, psd_one_sided[-2:0:-1]))
    return psd


def _freq_to_time(Y_synth: ndarray, n_samples_psd_one_sided: int) -> ndarray:
    n_samples_psd_one_sided = int(len(Y_synth) / 2)
    # Create random phases for all FFT terms.
    # The randomized phase needs to be symmetric in regards to the origin.
    phases = np.random.uniform(0, 2 * np.pi, (n_samples_psd_one_sided,))
    Y_synth[1 : n_samples_psd_one_sided + 1] *= np.exp(2j * phases)
    Y_synth[n_samples_psd_one_sided:] *= np.exp(-2j * phases[::-1])

    # Calculate the signal in the time domain
    y_synth = scipy.fft.ifft(Y_synth).real
    # The signal needs to be scaled by the number of samples
    # of the one sided PSD
    y_synth *= n_samples_psd_one_sided
    return y_synth


class NoiseGenerator(Protocol):
    """A protocol for noise generators."""

    def generate(self, n_samples: int) -> ndarray:
        """Generate noise.

        Args:
            n_samples: The number of samples to generate.

        Returns:
            The generated noise.
        """
        ...


class PSDNoiseGenerator(NoiseGenerator):
    """Synthesize random noise from a given PSD."""

    def __init__(self, psd_one_sided: ndarray, standard_deviation: float) -> None:
        """Initialize the noise synthesizer.

        Args:
            psd_one_sided: The one-sided power spectral density of the noise.
            standard_deviation: The desired standard deviation of the noise.
        """
        self.n_samples_psd_one_sided = len(psd_one_sided)
        psd = _get_two_sided_psd(psd_one_sided)
        self.freq_spectrum = np.sqrt(psd)
        self.freq_spectrum_resampled = self.freq_spectrum
        self._standard_deviation = standard_deviation

    def generate(self, n_samples: int):
        """Synthesize a given number of noise samples.

        Args:
            n_samples: The number of samples to generate.

        Returns:
            The signal in the time domain.
        """
        if len(self.freq_spectrum_resampled) != n_samples:
            self.freq_spectrum_resampled = signal.resample(
                self.freq_spectrum, n_samples
            )
        noise = _freq_to_time(
            self.freq_spectrum_resampled, self.n_samples_psd_one_sided
        )

        # Scale to desired std deviation
        noise = noise / np.std(noise) * self._standard_deviation
        # Center around zero
        noise = noise - np.mean(noise)

        return noise


class GaussianNoiseGenerator(NoiseGenerator):
    """Colored noise generator."""

    def __init__(
        self,
        beta: float,
        standard_deviation: float,
        fmin: float,
        random_seed: Optional[int],
    ) -> None:
        """Initialize the noise generator.

        Random noise will be pre-generated on all channels.

        Args:
            beta: The power-spectrum of the generated noise is proportional to
                (1 / f)**beta.
            standard_deviation: The desired standard deviation of the noise.
            fmin: Low-frequency cutoff. `fmin` is normalized frequency and the range is
                between `1/samples` and `0.5`, which corresponds to the Nyquist
                frequency.
            random_seed: The random seed to use. Use a fixed seed
                for reproducible results.
        """
        self._beta = beta
        self._standard_deviation = standard_deviation
        self._fmin = fmin
        self._random_seed = random_seed

    def generate(self, n_samples: int) -> ndarray:
        """Generate colored noise.

        Args:
            n_samples: The number of samples to generate.

        Returns:
            The generated noise.
        """
        noise = colorednoise.powerlaw_psd_gaussian(
            self._beta, n_samples, fmin=self._fmin, random_state=self._random_seed
        )
        noise *= self._standard_deviation
        return noise


class ZeroNoiseGenerator(NoiseGenerator):
    """Zero amplitude noise generator."""

    def generate(self, n_samples: int) -> ndarray:
        """Generate zero amplitude noise.

        Args:
            n_samples: The number of samples to generate.

        Returns:
            The generated noise.
        """
        return np.zeros((n_samples,))
