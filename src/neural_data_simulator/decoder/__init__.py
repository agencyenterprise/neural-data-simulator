"""This package contains an example implementation of a neural decoder.

The decoder is a software component that takes neural raw data as input
and outputs behavioral data.

The decoder entry point is the :meth:`decoder.run_decoder.run` method.
Upon start, the configuration is parsed, the decoder connects to the
LSL input raw data stream and creates an LSL outlet to write the decoded
data to. Decoding is an iterative process that runs in a loop. On each
iteration, a chunk of raw data is read from the LSL input stream and
added to a buffer. The buffer is used to accumulate raw data so that
the decoded chunk size can be constant. Once the buffer contains enough
data according to the predetermined decoding window size, the decoder
will count the number of spikes in each channel and estimate the spike
rates. The spike rates are then used to predict the behavioral data
using a pre-trained model. The predicted data is then written to the
LSL output stream.
"""
