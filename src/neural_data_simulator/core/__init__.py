"""Neural Data Simulator core package.

The neural data simulator, or NDS, aims to create a system that can generate spiking
data from behavioral data (e.g. cursor movement, arm kinematics, etc) in real-time.

The structure of this package is as follows:
 - :mod:`encoder` contains the `encoder` implementation.
 - :mod:`ephys_generator` currently hosts all classes that are used to generate
   spikes from spike rates.
 - :mod:`filters` contains all the filter implementations used for signal
   processing.
 - :mod:`health_checker` hosts utilities that monitor the data processing time
   consistency. It prints messages to the console when the data processing time
   is increasing or is taking longer than expected.
 - :mod:`inputs` contains various types of inputs that can be used by NDS,
   mainly :class:`inputs.SamplesInput` and :class:`inputs.LSLInput`.
 - :mod:`models` is meant to host model implementations that can be used by the
   encoder to convert behavior data into spike rates.
 - :mod:`outputs` contains classes that NDS uses to stream data out, mainly
   :class:`outputs.FileOutput`, :class:`outputs.ConsoleOutput` and
   :class:`outputs.LSLOutputDevice`.
 - :mod:`runner` is where the function that runs the encoder pipeline is
   implemented.
 - :mod:`samples` contains the :class:`samples.Samples` class, which is used to
   represent data in NDS.
 - :mod:`settings` contains the data model used to parse and validate the
   config.
 - :mod:`timing` contains the :class:`timing.Timer` class, which is used by
   every script that processes data at regular intervals.
"""
