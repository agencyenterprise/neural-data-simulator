"""Neural Data Simulator core package.

The neural data simulator, or NDS, aims to create a system that can generate
spiking data from behavioral data (e.g. cursor movement, arm kinematics, etc) in
real-time.

The structure of this package is as follows:
 - :mod:`neural_data_simulator.core` contains the core NDS implementation,
    including the encoder (also known as a simulator or generator).
 - :mod:`neural_data_simulator.decoder` implements decoders that can decode
   behavioral variables from neural activity.
 - :mod:`neural_data_simulator.tasks` implements tasks that can be used with the
   encoder.
 - :mod:`neural_data_simulator.recorder` implements an extension to write data
   to file.
 - :mod:`neural_data_simulator.streamer` implements an extension to stream data
   from file.
 - :mod:`neural_data_simulator.util` contains various data structures and
   utility functions that are used throughout NDS.
 - :mod:`neural_data_simulator.scripts` hosts many of the entry points for the
   scripts that are exposed to the user.
 - :mod:`neural_data_simulator.plugins` includes additional plugins for NDS.
"""
