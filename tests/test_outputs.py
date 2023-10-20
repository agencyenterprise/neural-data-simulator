"""Tests for outputs.py module."""
from unittest import mock

import numpy as np
import pytest

import neural_data_simulator.core.outputs
from neural_data_simulator.core.samples import Samples


@pytest.fixture
def mock_time(monkeypatch):
    """Override the time module with a mock that we can control."""
    time_mock = mock.Mock()
    monkeypatch.setattr(neural_data_simulator.core.inputs, "time", time_mock)
    return time_mock


@pytest.fixture
def samples_to_send():
    """Create data samples for tests."""
    return Samples(
        data=np.array([[1.0, 2.0]]),
        timestamps=np.array([1.2]),
    )


class TestConsoleOutput:
    """Tests for ConsoleOutput class."""

    def test_connect(self):
        """Test output can be connected using a context manager."""
        data_output = neural_data_simulator.core.outputs.ConsoleOutput(channel_count=1)
        data_output.connect()

    def test_send(self, samples_to_send, capsys):
        """Test if when `send` is called, the samples are printed to console."""
        data_output = neural_data_simulator.core.outputs.ConsoleOutput(channel_count=2)
        output_samples = data_output.send(samples_to_send)
        captured = capsys.readouterr()
        assert captured.out == "[[1.2 1.  2. ]]\n"
        assert output_samples == samples_to_send

    def test_send_data_with_wrong_shape(self, samples_to_send):
        """Test an exception is raised when the data has the wrong shape.

        The data has 2 channels, but the output only has 1 channel.
        """
        data_output = neural_data_simulator.core.outputs.ConsoleOutput(channel_count=1)
        with pytest.raises(ValueError):
            data_output.send(samples_to_send)


class TestFileOutput:
    """Tests for FileOutput class."""

    def test_connected(self):
        """Test output can be connected and output file is opened."""
        data_output = neural_data_simulator.core.outputs.FileOutput(channel_count=1)
        data_output.connect()
        assert data_output.file is not None
        assert not data_output.file.closed

    def test_disconnected(self):
        """Test that the file is closed when we context manager is exited."""
        data_output = neural_data_simulator.core.outputs.FileOutput(channel_count=1)
        data_output.connect()
        data_output.disconnect()
        assert data_output.file.closed

    def test_send(self, samples_to_send, tmpdir):
        """Test that sample is written to file when `send` is called."""
        file = tmpdir.join("output.csv")
        data_output = neural_data_simulator.core.outputs.FileOutput(
            channel_count=2, file_name=str(file)
        )
        data_output.connect()
        output_chunk = data_output.send(samples_to_send)
        data_output.disconnect()
        assert file.read() == "1.200000,1.000000,2.000000\n"
        assert output_chunk == samples_to_send

    def test_send_data_with_wrong_shape(self, samples_to_send):
        """Test an exception is raised when the data has the wrong shape.

        The data has 2 channels, but the output only has 1 channel.
        """
        data_output = neural_data_simulator.core.outputs.FileOutput(channel_count=1)
        with pytest.raises(ValueError):
            data_output.send(samples_to_send)


@pytest.fixture
def mock_lsl_outlet(monkeypatch):
    """Override the time module with a mock that we can control.

    Returns the mock so its behavior can be customized.
    """
    lsl_outlet = mock.Mock()
    pylsl_mock = mock.Mock()
    monkeypatch.setattr(neural_data_simulator.core.outputs, "pylsl", pylsl_mock)
    pylsl_mock.StreamOutlet = lsl_outlet
    pylsl_mock.resolve_streams = lambda: []
    return lsl_outlet


class TestLSLOutputDevice:
    """Tests for LSLOutputDevice class."""

    @property
    def fake_stream_config(self):
        """Get a generic stream config for tests."""
        stream_config = neural_data_simulator.core.outputs.StreamConfig(
            name="Test",
            type="behavior",
            source_id="a-test-fake",
            acquisition={
                "manufacturer": "Blackrock Neurotech",
                "model": "Simulated",
                "instrument_id": 0,
            },
            sample_rate=50,
            channel_format="int16",
            channel_labels=["1", "2"],
        )
        return stream_config

    def test_send_before_connection_raises_error(self, samples_to_send):
        """Test that sending samples before opening a connection raises an error."""
        lsl_output = neural_data_simulator.core.outputs.LSLOutputDevice(
            self.fake_stream_config
        )
        with pytest.raises(ConnectionError):
            lsl_output.send(samples_to_send)

    def test_send(self, samples_to_send, mock_lsl_outlet):
        """Test that samples are pushed to the LSL outlet."""
        lsl_output = neural_data_simulator.core.outputs.LSLOutputDevice(
            self.fake_stream_config
        )
        lsl_output.connect()
        mock_lsl_outlet.mock_calls = []
        lsl_output.send(samples_to_send)
        # using mock.ANY because numpy array doesn't like ==
        assert mock_lsl_outlet.mock_calls == [mock.call().push_sample(mock.ANY, 1.2)]
        assert list(mock_lsl_outlet.mock_calls[0][1][0]) == [1.0, 2.0]

    def test_send_no_data(self, mock_lsl_outlet):
        """Test that nothing is pushed to the LSL outlet if there is no data."""
        lsl_output = neural_data_simulator.core.outputs.LSLOutputDevice(
            self.fake_stream_config
        )
        lsl_output.connect()
        mock_lsl_outlet.mock_calls = []
        lsl_output.send(
            Samples(
                data=np.array([]),
                timestamps=np.array([]),
            )
        )
        assert mock_lsl_outlet.mock_calls == []

    def test_send_data_with_wrong_shape(self):
        """Test an exception is raised when the data has the wrong shape.

        The data has 2 channels, but the output only has 1 channel.
        """
        data_output = neural_data_simulator.core.outputs.LSLOutputDevice(
            self.fake_stream_config
        )
        with pytest.raises(ValueError):
            data_output.send(
                Samples(timestamps=np.array([1.2]), data=np.array([[1.0]]))
            )

    def test_send_as_chunk_with_wrong_shape(self):
        """Test an exception is raised when the data has the wrong shape.

        The data has 2 channels, but the output only has 1 channel.
        """
        data_output = neural_data_simulator.core.outputs.LSLOutputDevice(
            self.fake_stream_config
        )
        with pytest.raises(ValueError):
            data_output.send_as_chunk(data=np.array([[1.0]]))
