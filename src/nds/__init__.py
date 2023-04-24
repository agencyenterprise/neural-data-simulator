"""Neural Data Simulator main package.

The neural data simulator, or NDS, aims to create a system that can generate spiking
data from behavioral data (e.g. cursor movement, arm kinematics, etc) in real-time.

The structure of this package is as follows:
 - :mod:`nds.encoder` contains the `encoder` implementation.
 - :mod:`nds.ephys_generator` currently hosts all classes that are used to generate \
    spikes from spike rates.
 - :mod:`filters` contains all the filter implementations used for signal processing.
 - :mod:`health_checker` hosts utilities that monitor the data processing time \
    consistency. It prints messages to the console when the data processing time is \
        increasing or is taking longer than expected.
 - :mod:`nds.inputs` contains various types of inputs that can be used by NDS, mainly \
    :class:`nds.inputs.SamplesInput` and :class:`nds.inputs.LSLInput`.
 - :mod:`nds.models` is meant to host model implementations that can be used by the \
    encoder to convert behavior data into spike rates.
 - :mod:`nds.outputs` contains classes that NDS uses to stream data out, mainly \
    :class:`nds.outputs.FileOutput`, :class:`nds.outputs.ConsoleOutput` and \
        :class:`nds.outputs.LSLOutputDevice`.
 - :mod:`nds.runner` is where the function that runs the encoder pipeline is \
    implemented.
 - :mod:`nds.samples` contains the :class:`nds.samples.Samples` class, which is used \
    to represent data in NDS.
 - :mod:`nds.settings` contains the data model used to parse and validate the config.
 - :mod:`nds.streamers` currently hosts the :class:`nds.streamers.LSLStreamer` class, \
    which can be used to output :class:`nds.samples.Samples` to an LSL outlet.
 - :mod:`nds.timing` contains the :class:`nds.timing.Timer` class, which is used by \
    every script that processes data at regular intervals.

In addition, nds hosts the following subpackages:
 - :mod:`nds.util` contains various data structures and utility functions that are \
    used throughout NDS.
 - :mod:`nds.scripts` hosts the entry points for the scripts that are exposed to the \
    user.
"""
