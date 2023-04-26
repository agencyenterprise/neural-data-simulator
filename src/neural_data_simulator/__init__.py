"""Neural Data Simulator main package.

The neural data simulator, or NDS, aims to create a system that can generate spiking
data from behavioral data (e.g. cursor movement, arm kinematics, etc) in real-time.

The structure of this package is as follows:
 - :mod:`neural_data_simulator.encoder` contains the `encoder` implementation.
 - :mod:`neural_data_simulator.ephys_generator` currently hosts all classes that are
   used to generate spikes from spike rates.
 - :mod:`filters` contains all the filter implementations used for signal processing.
 - :mod:`health_checker` hosts utilities that monitor the data processing time
   consistency. It prints messages to the console when the data processing time is
   increasing or is taking longer than expected.
 - :mod:`neural_data_simulator.inputs` contains various types of inputs that can be used
   by NDS, mainly :class:`neural_data_simulator.inputs.SamplesInput` and
   :class:`neural_data_simulator.inputs.LSLInput`.
 - :mod:`neural_data_simulator.models` is meant to host model implementations that can
   be used by the encoder to convert behavior data into spike rates.
 - :mod:`neural_data_simulator.outputs` contains classes that NDS uses to stream data
   out, mainly :class:`neural_data_simulator.outputs.FileOutput`,
   :class:`neural_data_simulator.outputs.ConsoleOutput` and
   :class:`neural_data_simulator.outputs.LSLOutputDevice`.
 - :mod:`neural_data_simulator.runner` is where the function that runs the encoder
   pipeline is implemented.
 - :mod:`neural_data_simulator.samples` contains the
   :class:`neural_data_simulator.samples.Samples` class, which is used to represent
   data in NDS.
 - :mod:`neural_data_simulator.settings` contains the data model used to parse and
   validate the config.
 - :mod:`neural_data_simulator.streamers` currently hosts the
   :class:`neural_data_simulator.streamers.LSLStreamer` class,
   which can be used to output :class:`neural_data_simulator.samples.Samples` to an
   LSL outlet.
 - :mod:`neural_data_simulator.timing` contains the
   :class:`neural_data_simulator.timing.Timer` class, which is used by
   every script that processes data at regular intervals.

In addition, neural_data_simulator hosts the following subpackages:
 - :mod:`neural_data_simulator.util` contains various data structures and utility
   functions that are used throughout NDS.
 - :mod:`neural_data_simulator.scripts` hosts the entry points for the scripts that
   are exposed to the user.
"""
