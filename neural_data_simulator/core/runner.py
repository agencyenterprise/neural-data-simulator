"""Runner that uses a timer object to run the simulation."""

import contextlib
import logging
from typing import Iterator, Optional, Protocol

logger = logging.getLogger(__name__)


class Encoder(Protocol):
    """Protocol of an Encoder class.

    A python protocol (`PEP-544 <https://peps.python.org/pep-0544/>`_) works in
    a similar way to an abstract class.
    The :meth:`__init__` method of this protocol should never be called as
    protocols are not meant to be instantiated. An :meth:`__init__` method
    may be defined in a concrete implementation of this protocol if needed.
    """

    def iterate(self) -> None:
        """Iterate over steps of a simulation."""
        ...

    @contextlib.contextmanager
    def connect(self) -> Iterator[None]:
        """Connect to a encoder."""
        ...


class Timer(Protocol):
    """Protocol for a Timer class.

    A python protocol (`PEP-544 <https://peps.python.org/pep-0544/>`_) works in
    a similar way to an abstract class.
    The :meth:`__init__` method of this protocol should never be called as
    protocols are not meant to be instantiated. An :meth:`__init__` method
    may be defined in a concrete implementation of this protocol if needed.
    """

    def start(self) -> None:
        """Start timer."""
        ...

    def wait(self) -> None:
        """Wait appropriate time."""
        ...

    def total_elapsed_time(self) -> float:
        """Get total time since start.

        Returns:
            Total time since start in seconds.
        """
        ...


def _is_simulation_complete(
    timer: Timer, total_seconds_of_simulation: Optional[int] = None
) -> bool:
    if not total_seconds_of_simulation:
        return False
    else:
        return timer.total_elapsed_time() > total_seconds_of_simulation


def run(
    encoder: Encoder,
    timer: Timer,
    total_seconds_of_simulation: Optional[int] = None,
):
    """Loop over the provided encoder using a timer.

    Connects to all devices in the encoder before starting the timer. Using
    the timer, the loop should execute the encoder once per timer period.
    CTR+C can be used to stop the loop at any time, after which all devices
    will be disconnected and the function will return.

    Args:
        encoder: Encoder object to be executed periodically. It can be
          connected to through a context manager `connect` method and can be iterated
          using the `iterate` method.
        timer: Timer object that can be started with `start`, waits (sleep) for
          the necessary time using `wait`, and can return the total elapsed time since
          `start` through `total_elapsed_time` method.
        total_seconds_of_simulation: Total time to run the simulation (in seconds)
          or None if it should run indefinitely (until CTR+C is pressed). Defaults to
          None.
    """
    with encoder.connect():
        logger.info("Starting loop. Press CTRL+C at any time to exit loop.")
        timer.start()

        while not _is_simulation_complete(timer, total_seconds_of_simulation):
            timer.wait()
            encoder.iterate()
