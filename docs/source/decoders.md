# Decoders

The decoder is responsible for predicting behavior from electrophysiology raw data. In the BCI closed loop paradigm, the decoder is consuming the neural data and its output is used to control an external device (e.g., a cursor on a screen) in order to perform a task.

NDS comes with a basic, yet complete, decoder implementation that can be used as a template to guide your own development. The NDS decoder input is the raw data stream from the `ephys generator` and the output is the predicted cursor movement in the horizontal and vertical directions.

## Provided decoder model

The complete decoder implementation including the signal processing (spike detection and spike rate estimation) is contained in the {class}`decoder.decoders.Decoder` class. This class is using a linear regression model that predicts cursor movement from spike rates. See the [Train models for the encoder and decoder](auto_examples/plot_train_encoder_and_decoder_model) example for more details on how the model was trained.

The {class}`decoder.decoders.Decoder` class can also be used with a custom decoder model implementation that conforms to the {class}`decoder.decoders.DecoderModel` protocol.

## Configuring and running the included decoder

To configure the decoder, change the file `settings_decoder.yaml`, which is located by default in the `$HOME/.nds/` folder.  To use a specific config file, specify the config directory (`--config-dir`) and config file-name (`--config-name`) flags. For example:
```
decoder --config-dir $HOME/.nds/ --config-name settings
```

Upon start, the decoder connects to the LSL input raw data stream and creates an LSL outlet to write the predicted behavior to. If the input stream cannot be found, the decoder will not be able to start, therefore make sure that the `ephys generator` is running before starting the decoder.

To use a different model file for predicting velocities, change the following config:

```
decoder:
  model_file: "sample_data/session_4_simple_decoder.joblib"
```

The threshold for spike detection can be adjusted by updating the config:

```
decoder:
  spike_threshold: -200 # uV
```

To start the decoder, complete the [installation](installation.md) with the `[extras]` option, then run the script:

```
decoder
```

## Integrating a new decoder

If the included [Decoder](#decoder.decoders.Decoder) is not well suited for your use case and training a new decoder model by following the [example](auto_examples/plot_train_encoder_and_decoder_model) is not a good option, you can create your own decoder and integrate it into the NDS closed loop. For a seamless integration your decoder should satisfy the following requirements:

- provide an LSL outlet named `NDS-Decoder`. By default, the GUI is configured to read data from this outlet.
- output at `50 Hz`. By default, the GUI loop is running at 50 Hz.
- output data is an array with shape (n_samples, 2) where the 2 columns correspond to the velocity in the horizontal and vertical direction.

A new decoder can read data from one of the different LSL streams that NDS provides by default:

- `NDS-RawData`: the raw spiking data stream.
- `NDS-LFPData`: the local field potentials stream.
- `NDS-SpikeEvents`: the stream containing spike events for each unit.
- `NDS-SpikeRates`: the stream containing spike rates for each channel.

```{note}
When adding a new decoder to the loop, ensure that the default decoder is not running in order to prevent an LSL outlet name conflict. To achieve this change the `run-closed-loop` Makefile target removing the `poetry run decoder` command.
```

For example, a decoder that reads based on the `NDS-SpikeRates` stream could be implemented as follows:

```
import os
import time

import joblib
import numpy as np
import pylsl
from neural_data_simulator.util.runtime import get_sample_data_dir


def get_lsl_outlet():
    stream_info = pylsl.StreamInfo(
        name="NDS-Decoder",
        type="behavior",
        channel_count=2,
        nominal_srate=50,
        channel_format="float32",
        source_id="centerout_behavior",
    )
    channels_info = stream_info.desc().append_child("channels")
    for ch_name in ["vel_x", "vel_y"]:
        ch = channels_info.append_child("channel")
        ch.append_child_value("label", ch_name)

    return pylsl.StreamOutlet(stream_info)


def get_lsl_inlet():
    stream_infos = pylsl.resolve_byprop("name", "NDS-SpikeRates")
    if len(stream_infos) > 0:
        return pylsl.StreamInlet(stream_infos[0])
    else:
        raise ConnectionError("Inlet could not find requested LSL stream.")


def main():
    loaded_decoder = joblib.load(
        os.path.join(get_sample_data_dir(), "session_4_simple_decoder.joblib")
    )
    n_channels = 190
    decoder_interval = 20 * 1e6  # nanoseconds

    lsl_outlet = get_lsl_outlet()
    lsl_inlet = get_lsl_inlet()
    last_run_time = time.perf_counter_ns()

    while True:
        time_now = time.perf_counter_ns()
        elapsed_time_ns = time_now - last_run_time
        if elapsed_time_ns >= decoder_interval:
            samples, _ = lsl_inlet.pull_chunk()
            if len(samples) > 0:
                spike_rates = np.array(samples)[0]
                velocities = loaded_decoder.predict(spike_rates.reshape(-1, n_channels))
                lsl_outlet.push_sample(velocities[0])
            last_run_time = time_now

        else:
            time.sleep(0.0001)


if __name__ == "__main__":
    main()

```

This decoder is not practical because it is simply reversing the output of the encoder, but it can serve as a starting point for your own decoder since it satisfies the requirements mentioned above.
