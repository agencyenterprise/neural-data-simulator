"""A collection of inputs that can be used by NDS."""
import abc
from typing import Optional, Protocol

import numpy as np

from neural_data_simulator.core.samples import Samples


class Input(abc.ABC):
    """Represents an input that can be used to consume data from.

    This can be an interface for a joystick, a behavior data generator, a data
    streamer that loads data from disk, etc.
    Each `read` should return all newly available data since the last `read` call.
    """

    @abc.abstractmethod
    def read(self) -> Samples:
        """Read available data."""
        pass

    @abc.abstractmethod
    def connect(self) -> None:
        """Connect to input."""
        pass

    def disconnect(self) -> None:
        """Disconnect from input. The default implementation does nothing."""
        pass


class SpikeRateInput(Protocol):
    """An abstract input that can be used to read spike rates.

    A python protocol (`PEP-544 <https://peps.python.org/pep-0544/>`_) works in
    a similar way to an abstract class.
    The :meth:`__init__` method of this protocol should never be called as
    protocols are not meant to be instantiated. An :meth:`__init__` method
    may be defined in a concrete implementation of this protocol if needed.
    """

    @property
    def channel_count(self) -> int:
        """Get the number of input channels.

        Returns:
            The input channel count.
        """
        ...

    def read(self) -> Optional[np.ndarray]:
        """Read spike rates, one per channel.

        Note: above user can call Input.read() but just return the latest Sample

        Returns:
            An array of spike rates with shape (n_units,) or None if no samples
            are available.
        """
        ...
