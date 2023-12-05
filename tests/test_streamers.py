"""Test streamers.py module."""

import numpy as np
import pytest

from neural_data_simulator.core.outputs import lsl_output
from neural_data_simulator.core.samples import Samples
from neural_data_simulator.streamer.streamers import LSLStreamer


class StreamOutletFake:
    """Fake pylsl.StreamOutlet."""

    def __init__(self, *args, **kwargs):
        """Fake init method."""
        self.pushed_sample_data = np.zeros(shape=(0, 2))
        self.pushed_chunk_data = np.zeros(shape=(0, 2))

    def connect(self):
        """Fake connect method."""
        pass

    def push_chunk(self, data, timestamp):
        """Fake push_chunk method. Saves passed data for later validation."""
        self.pushed_chunk_data = np.vstack((self.pushed_chunk_data, data))
        # `timestamp` arg-type can be flexible, so we ignore it for now
        _ = timestamp

    def push_sample(self, data, timestamp=None):
        """Fake push_sample method. Saves passed data for later validation."""
        self.pushed_sample_data = np.vstack((self.pushed_sample_data, data))


def get_stream_config(sample_rate):
    """Get a generic stream config for tests."""
    stream_config = lsl_output.StreamConfig(
        name="Test",
        type="behavior",
        source_id="a-test-fake",
        acquisition={
            "manufacturer": "Blackrock Neurotech",
            "model": "Simulated",
            "instrument_id": 0,
        },
        sample_rate=sample_rate,
        channel_format="int16",
        channel_labels=["1", "2"],
    )
    return stream_config


@pytest.fixture(autouse=True)
def fake_lsl_outlet(monkeypatch):
    """Override the pylsl.StreamOutlet with a fake."""
    monkeypatch.setattr(
        lsl_output.pylsl,
        "StreamOutlet",
        StreamOutletFake,
    )


class TestLSLStreamer:
    """Test LSLStreamer class."""

    def test_stream_samples(self):
        """Test that samples are forwarded to the LSL outlet by the streamer."""
        samples = Samples(timestamps=np.array([0, 1]), data=np.array([[2, 3], [4, 5]]))

        regular_stream_output = lsl_output.LSLOutputDevice(
            get_stream_config(sample_rate=50)
        )
        irregular_stream_output = lsl_output.LSLOutputDevice(
            get_stream_config(sample_rate=0)
        )
        streamer = LSLStreamer(
            [regular_stream_output, irregular_stream_output],
            [samples, samples],
            10,
            False,
        )
        regular_stream_output.connect()
        irregular_stream_output.connect()
        streamer.stream()
        fake_regular_lsl_outlet = regular_stream_output._outlet
        fake_irregular_lsl_outlet = irregular_stream_output._outlet

        np.testing.assert_array_equal(
            fake_regular_lsl_outlet.pushed_chunk_data,
            np.array([[2, 3], [4, 5]]),
        )
        np.testing.assert_array_equal(
            fake_irregular_lsl_outlet.pushed_chunk_data,
            np.array([[2, 3], [4, 5]]),
        )
