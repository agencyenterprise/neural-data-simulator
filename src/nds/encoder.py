"""This module contains the `Encoder` implementation."""
import contextlib
import logging
from typing import Iterator, Optional, Protocol, runtime_checkable

from nds.inputs import Input
from nds.models import EncoderModel
from nds.outputs import Output
from nds.samples import Samples
from nds.util.runtime import open_connection


@runtime_checkable
class Processor(Protocol):
    """Protocol for an encoder Processor class.

    A processor can be used to transform data, usually for the purpose of
    adapting it to match the requirements of the:

    - encoder model: in this case the processor is called a preprocessor.
    - consumer of the encoder output (spike rates): in this case the processor
      represents a postprocessor.

    A python protocol (`PEP-544 <https://peps.python.org/pep-0544/>`_) works in
    a similar way to an abstract class.
    The :meth:`__init__` method of this protocol should never be called as
    protocols are not meant to be instantiated. An :meth:`__init__` method
    may be defined in a concrete implementation of this protocol if needed.
    """

    def execute(self, data: Samples) -> Samples:
        """Execute processing on the samples input data.

        Args:
            data: Input data to process.

        Returns:
            Data samples after processing.
        """
        ...


class Encoder:
    """Encoder class implementation.

    It manages the data through all the necessary steps to convert from behavior data
    into spiking data. These steps currently include an optional preprocessor, the
    model transformation and an optional postprocessor.
    """

    def __init__(
        self,
        *,
        input_: Input,
        preprocessor: Optional[Processor],
        model: EncoderModel,
        postprocessor: Optional[Processor],
        output: Output,
    ):
        """Initialize the Encoder class.

        Args:
            input_: a class that implements reading one or multiple
                samples with the `read` method, and can be connected to
                through a context manager.
            preprocessor: optional processor to transform the
                samples before they are passed to the model.
            model: a class that can convert samples of behavior data into
                samples of spike rates for each call of the `encode` method.
            postprocessor: optional processor to transform the
                samples that are returned by the model.
            output: a class that can take one or multiple samples for each call
                of the `send` method, and can be connected to through a context manager.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Constructing Encoder")

        self.input = input_
        self.output = output
        self.preprocessor = preprocessor
        self.model = model
        self.postprocessor = postprocessor

    def iterate(self) -> None:
        """Move samples through all the stages of the Encoder.

        That is: behavior samples input -> preprocessing -> encoding -> postprocessing
        -> spike rates samples output.
        """
        samples = self.input.read()
        if len(samples):
            if self.preprocessor:
                samples = self.preprocessor.execute(samples)
            samples = self.model.encode(samples)
            if self.postprocessor:
                samples = self.postprocessor.execute(samples)
            self.output.send(samples)

    @contextlib.contextmanager
    def connect(self) -> Iterator[None]:
        """Connect to both input and output.

        Yields:
            Yields after both connections are established.
        """
        with open_connection(self.input), open_connection(self.output):
            yield
