"""Classes that implement a model for encoding neural data from behavior data."""
from typing import Protocol, runtime_checkable

from neural_data_simulator.samples import Samples


@runtime_checkable
class EncoderModel(Protocol):
    """Protocol for an `Encoder` model.

    Classes that conform to this protocol can be used by the
    :class:`neural_data_simulator.encoder.Encoder` to convert behavioral input data into
    spiking rate data.

    The Encoder processes data in chunks represented as
    :class:`neural_data_simulator.samples.Samples`.
    One chunk may contain several behavioral data points (n_samples) across
    multiple axes (n_axes). The Encoder calls the EncoderModel's :meth:`encode` method
    for each chunk in order to transform the behavioral data into spiking rates
    (n_samples) across multiple units (n_units).

    A python protocol (`PEP-544 <https://peps.python.org/pep-0544/>`_) works in
    a similar way to an abstract class.
    The :meth:`__init__` method of this protocol should never be called as
    protocols are not meant to be instantiated. An :meth:`__init__` method
    may be defined in a concrete implementation of this protocol if needed.
    """

    def encode(self, data: Samples) -> Samples:
        """Encode behavior into spiking rates.

        Args:
            data: Behavioral data as :class:`neural_data_simulator.samples.Samples`.
              For example, in case of modeling velocities in a horizontal and vertical
              direction (2 axes), the data is a 2D array with shape (n_samples, 2).

        Returns:
            Spiking rates as :class:`neural_data_simulator.samples.Samples`.
            The spiking rates are represented as a 2D array with shape
            (n_samples, n_units).
        """
        ...
