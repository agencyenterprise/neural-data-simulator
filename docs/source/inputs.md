# Behavioral Input

The [encoder](encoder.md) requires a stream of behavioral data as input. It currently supports 2 different data sources:

1. an [LSL stream](configuring.md#behavior-lsl-stream) of real-time behavior data.
2. playback of a numpy data file containing behavior data that was recorded with a test subject or artificially generated.

Ideally the input data matches the expectations of the [encoder](encoder.md) (zero mean and standard deviation of 1).
If the input data does not meet this requirement, a [preprocessor](encoder.md#preprocessor) can be used to transform the data to work with the encoder.

## Lab Streaming Layer data source

To use an LSL stream as the input to the encoder, ensure that the encoder input section of the configuration file is set to [LSL](configuring.md#behavior-lsl-stream) and that the `stream_name` parameter matches that of your running application. Some examples of LSL applications you can try:

- the [`streamer` application](utilities.md#streamer), which can play back a file over LSL;
- the provided example [`center_out_reach` application](tasks.md#provided-task), which should work with the encoder using default settings;
- you can use any LSL stream with continuous data
  - most such solutions will require encoder preprocessing to shape and scale the streams to match encoder input expectations.
  - For example, run the official LSL [Gamepad App](https://github.com/labstreaminglayer/App-Gamepad/releases). To use the data from the gamepad as input to NDS, modify the [encoder configuration](configuring.md#behavior-lsl-stream) `stream_name` to "Gamepad", and enable the custom preprocessor by uncommenting the preprocessor line and modify its value to `'plugins/gamepad_preprocessor.py'`.

## File data source

To use a file as the input source, ensure that the encoder input section of the configuration file is set to [file](configuring.md#prerecorded-behavior-file) and the associated parameters are set accordingly.

The data must be stored in a numpy file and contain at least two arrays:

1. a 1D array of timestamps where each element corresponds to a sample.
2. a 2D array of samples where each row corresponds to a sample and each column to a channel.

For example, if 2 samples were recorded from 3 channels the `data` and `timestamps` array could look like:

```python
timestamps = [0.0, 0.02]
data = [[1, 2, 3],[4, 5, 6]]
```

If you generate such an array, you may save it using the following snippet:

```python
import numpy as np

np.savez('rec_data.npz', data=data, timestamps=timestamps)
```

The name of the variables holding the arrays in the numpy file do not need to be `data` and `timestamps`, you can specify custom names in the [`settings.yaml`](configuring.md) file.
