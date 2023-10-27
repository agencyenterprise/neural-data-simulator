"""Test input.py module."""
import time
from unittest import mock

import numpy as np
import pylsl
import pytest

from neural_data_simulator.core.inputs import samples_input
from neural_data_simulator.core.inputs import lsl_input
from neural_data_simulator.core.samples import Samples


@pytest.fixture
def mock_time(monkeypatch):
    """Create a mock for the time class."""
    time_mock = mock.Mock()
    monkeypatch.setattr(samples_input, "time", time_mock)
    return time_mock


@pytest.fixture
def input_samples() -> Samples:
    """Create example samples with some data and timestamps for testing."""
    return Samples(data=np.arange(20).reshape(10, 2), timestamps=np.arange(10) * 0.02)


class TestSamplesInput:
    """Test samples input."""

    def test_referencing(self, input_samples):
        """Test if initialization is performed when explicitly called."""
        data_input = samples_input.SamplesInput(input_samples)
        data_input.set_reference_time_to_now()
        time.sleep(0.2)
        read_samples = data_input.read()
        assert read_samples == input_samples

    def test_read(self, input_samples, mock_time):
        """Test the read function.

        Validate that it is being called properly and that it is returning
        the expected samples.
        """
        data_input = samples_input.SamplesInput(input_samples)
        mock_time.time.side_effect = np.arange(60000.00, 60000.20, 0.02)
        data_input.set_reference_time_to_now()
        samples1 = data_input.read()
        samples2 = data_input.read()
        assert samples1 == Samples(data=np.array([[0, 1]]), timestamps=np.array([0.00]))
        assert samples2 == Samples(data=np.array([[2, 3]]), timestamps=np.array([0.02]))

    def test_read_without_referencing(
        self, input_samples, mock_time: mock.Mock, mocker
    ):
        """Test if the `set_reference_time_to_now` function is called.

        When `read` is called before `set_reference_time_to_now`,
        `set_reference_time_to_now` should be called automatically.
        """
        data_input = samples_input.SamplesInput(input_samples)
        mock_time.time.side_effect = [60000.00, 60000.00001]
        spy = mocker.spy(data_input, "set_reference_time_to_now")
        _ = data_input.read()
        assert spy.call_count == 1

    def test_read_between_timestamps(self, input_samples, mock_time: mock.Mock):
        """Test that no sample is returned when appropriate.

        If `read` is called before the time of the next sample, no sample should be
        returned.
        """
        data_input = samples_input.SamplesInput(input_samples)
        mock_time.time.side_effect = [60000.00, 60000.20, 60000.20001]
        _ = data_input.read()
        samples = data_input.read()
        assert len(samples) == 0


class TestLSLInput:
    """Test LSL input."""

    @pytest.fixture(scope="class", autouse=True)
    def fake_behavior_outlet(self):
        """Set up a fake eeg outlet."""
        stream_info = pylsl.StreamInfo(
            name="Test",
            type="behavior",
            channel_count=2,
            nominal_srate=50,
            channel_format="int16",
            source_id="a-test-fake",
        )
        return pylsl.stream_outlet(stream_info)

    def get_test_input(self, stream_name="Test"):
        """Create a testing LSL input."""
        return lsl_input.LSLInput(stream_name, resolve_streams_wait_time=0.01)

    def test_connect(self):
        """Test connecting to the input."""
        lsl_input = self.get_test_input()
        lsl_input.connect()
        assert lsl_input._inlet is not None

    def test_disconnect(self):
        """Test disconnecting from the input."""
        lsl_input = self.get_test_input()
        lsl_input.connect()
        lsl_input.disconnect()
        assert lsl_input._inlet is None

    def test_get_info_before_connect(self):
        """Test getting the info before connecting to stream."""
        lsl_input = self.get_test_input()
        info = lsl_input.get_info()
        assert info is not None
        assert info.name == "Test"
        assert info.sample_rate == 50

    def test_get_info_from_inlet(self):
        """Test getting the info from the LSL inlet."""
        lsl_input = self.get_test_input()
        lsl_input.connect()
        info = lsl_input.get_info()
        assert info is not None
        assert info.name == "Test"
        assert info.sample_rate == 50

    def test_timeout_to_connect_and_desired_stream_unavailable(self):
        """Test timeout and stream not available.

        Because the default is to look for a stream name for a very long time, here we
        need to change the timeout to be able get the error for stream not found.
        """
        lsl_input = self.get_test_input("WrongName")
        lsl_input.set_connection_timeout(0.001)
        with pytest.raises(ConnectionError):
            with lsl_input.connect():
                time.sleep(1)

    def test_set_timeout_to_negative(self):
        """Test exception if timeout is set to a negative number."""
        lsl_input = self.get_test_input()
        with pytest.raises(ValueError):
            lsl_input.set_connection_timeout(-1)

    def test_set_timeout_to_zero(self):
        """Test exception if timeout is set to zero."""
        lsl_input = self.get_test_input()
        with pytest.raises(ValueError):
            lsl_input.set_connection_timeout(0)

    def test_connection_when_multiple_streams_are_available(self):
        """Test that we connect to the correct stream when multiple are available."""
        stream_info = pylsl.StreamInfo(
            name="WrongStream",
            type="behavior",
            channel_count=2,
            nominal_srate=50,
            channel_format="int16",
            source_id="a-test-fake",
        )
        _ = pylsl.stream_outlet(stream_info)
        lsl_input = self.get_test_input()
        lsl_input.connect()
        assert lsl_input._inlet.info().name() == "Test"

    def test_read_samples_before_connection_raises_error(self):
        """Test that reading samples before connection raises an error."""
        lsl_input = self.get_test_input()
        with pytest.raises(ConnectionError):
            lsl_input.read()

    @mock.patch("pylsl.stream_inlet.pull_chunk")
    def test_read_samples(self, reads_chunk):
        """Test reading a chunk from the LSL stream."""
        timestamps = np.arange(3).tolist()
        data = np.arange(6).reshape(3, 2)

        def pull_chunk(timeout, max_samples, dest_obj):
            dest_obj[:3] = data
            return None, timestamps

        reads_chunk.side_effect = pull_chunk
        lsl_input = self.get_test_input()
        lsl_input.connect()
        assert lsl_input.read() == Samples(np.array(timestamps), data)

    @mock.patch("pylsl.stream_inlet.pull_chunk")
    def test_read_empty_samples(self, reads_chunk):
        """Test reading when no data is available."""
        timestamps = []
        data = []
        reads_chunk.side_effect = [(data, timestamps)]
        lsl_input = self.get_test_input()
        lsl_input.connect()
        samples_read = lsl_input.read()
        assert samples_read == Samples(
            timestamps=np.array(timestamps), data=np.array(data)
        )
