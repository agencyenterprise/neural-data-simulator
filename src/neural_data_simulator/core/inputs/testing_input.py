"""Example input classes for testing."""

from typing import Optional

import numpy as np

from neural_data_simulator.core.inputs.api import SpikeRateInput


class SpikeRateTestingInput(SpikeRateInput):
    """A constant spike rate input that can be used for testing.

    Generates spike rates so that spikes are more likely to happen
    on channels of a higher order and less likely on channels of a lower order.
    The spike rate for a channel is always constant.
    """

    def __init__(self, n_channels: int, n_units: int):
        """Create a testing spike rate input.

        Args:
            n_channels: The number of input channels.
            n_units: The total number of units, which should be a multiple
                of the number of channels.
        """
        self.n_channels = n_channels
        self.n_units = n_units

    @property
    def channel_count(self) -> int:
        """Get the number of input channels.

        Returns:
            The input channel count.
        """
        return self.n_channels

    def read(self) -> Optional[np.ndarray]:
        """Read spike rates, one per channel.

        Returns:
            The array of testing spike rates with shape (n_units,).
            For example, if `n_channels = 50` and `n_units_per_channel = 1`, the
            spike rates will be constant and equal to:

            `[ 0.  2.  4.  6.  8. 10. 12. 14. 16. 18. 20. 22. 24. 26. 28. 30. 32. 34.
            36. 38. 40. 42. 44. 46. 48. 50. 52. 54. 56. 58. 60. 62. 64. 66. 68. 70.
            72. 74. 76. 78. 80. 82. 84. 86. 88. 90. 92. 94. 96. 98.]`
        """
        rates = np.arange(self.n_units) * 100 / self.n_units
        return rates.astype(int)
