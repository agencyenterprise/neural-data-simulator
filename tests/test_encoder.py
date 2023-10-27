"""Test encoder.py module."""
from unittest.mock import patch

import numpy as np

import neural_data_simulator.core.encoder as encoder
from neural_data_simulator.core.inputs import api as inputs
import neural_data_simulator.core.outputs.api as outputs
from neural_data_simulator.core.samples import Samples


class FakeInput(inputs.Input):
    """Create an input-like class for testing."""

    def read(self) -> Samples:
        """Read the available samples."""
        return Samples(timestamps=np.array([0.1]), data=np.array([[0.2, 0.3]]))

    def connect(self):
        """Connect to the input."""
        yield


class FakeOutput(outputs.Output):
    """Create an output-like class for testing."""

    @property
    def channel_count(self) -> int:
        """Return the number of samples for this output."""
        return 2

    def _send(self, samples):
        """Send samples."""
        pass

    def connect(self):
        """Connect to output."""
        yield


class FakeModel:
    """Create an encodermodel-like class for testing."""

    def encode(self, samples):
        """Encode behavior into spikes. For testing, just return the input samples."""
        return samples


class FakePreprocessor:
    """Create an preprocessor-like class for testing."""

    def execute(self, samples):
        """For testing, just return the input samples plus 1."""
        return Samples(timestamps=samples.timestamps, data=samples.data + 1)


class FakePostprocessor:
    """Create an postprocessor-like class for testing."""

    def execute(self, samples):
        """For testing, just return the input samples plus 2."""
        return Samples(timestamps=samples.timestamps, data=samples.data + 2)


class TestEncoder:
    """Test the encoder class."""

    @patch.object(FakeInput, "connect")
    @patch.object(FakeOutput, "connect")
    def test_connect(self, mock_output_connect, mock_input_connect):
        """Test encoder connection.

        For that we test that the `connect` method from the input and output are called.
        """
        input = FakeInput()
        output = FakeOutput()
        model = FakeModel()

        sim = encoder.Encoder(
            input_=input,
            output=output,
            model=model,
            preprocessor=None,
            postprocessor=None,
        )
        with sim.connect():
            mock_input_connect.assert_called_once()
            mock_output_connect.assert_called_once()

    @patch.object(FakeInput, "read")
    @patch.object(FakeModel, "encode")
    @patch.object(FakeOutput, "send")
    def test_iteration(self, mock_output_send, mock_model_encode, mock_input_read):
        """Test a encoder iteration.

        Test that the correct methods from the input, output, and model are called when
        the encoder is iterated.
        """
        input = FakeInput()
        output = FakeOutput()
        model = FakeModel()

        mock_input_read.return_value = Samples(
            timestamps=np.array([0.1]), data=np.array([[0.2, 0.3]])
        )
        sim = encoder.Encoder(
            input_=input,
            output=output,
            model=model,
            preprocessor=None,
            postprocessor=None,
        )

        sim.iterate()

        mock_input_read.assert_called_once()
        mock_model_encode.assert_called_once()
        mock_output_send.assert_called_once()

    @patch.object(FakeInput, "read")
    @patch.object(FakeOutput, "send")
    def test_iteration_with_pre_and_postprocessor(
        self, mock_output_send, mock_input_read
    ):
        """Test a encoder iteration.

        Test that the correct methods from the input, output, and model are called when
        the encoder is iterated.
        """
        input = FakeInput()
        output = FakeOutput()
        model = FakeModel()
        preprocessor = FakePreprocessor()
        postprocessor = FakePostprocessor()

        mock_input_read.return_value = Samples(
            timestamps=np.array([0.1]), data=np.array([[0.2, 0.3]])
        )
        sim = encoder.Encoder(
            input_=input,
            output=output,
            model=model,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
        )

        sim.iterate()

        mock_input_read.assert_called_once()
        mock_output_send.assert_called_once()
        mock_output_send.assert_called_with(
            Samples(timestamps=np.array([0.1]), data=np.array([[3.2, 3.3]]))
        )

    @patch.object(FakeInput, "read")
    @patch.object(FakeModel, "encode")
    @patch.object(FakeOutput, "send")
    def test_read_no_data(self, mock_output_send, mock_model_encode, mock_input_read):
        """Test that when no data is read from the input.

        The remaining of the simulation should not be executed.
        """
        input = FakeInput()
        output = FakeOutput()
        model = FakeModel()
        mock_input_read.return_value = Samples(
            timestamps=np.array([]), data=np.array([])
        )
        sim = encoder.Encoder(
            input_=input,
            output=output,
            model=model,
            preprocessor=None,
            postprocessor=None,
        )

        sim.iterate()

        mock_input_read.assert_called_once()
        mock_model_encode.assert_not_called()
        mock_output_send.assert_not_called()
