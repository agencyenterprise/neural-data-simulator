# flake8: noqa
"""
=========================================
Visualize Power Spectral Density (PSD)
=========================================

.. currentmodule:: neural_data_simulator

The goal of this example is to calculate and plot the Power Spectral Density (PSD) for a given channel
of the raw data.

By default, this script will download the data to be plotted from AWS S3. If you prefer to use
your own data, you can start the closed loop simulation in one terminal::

    make run-closed-loop

And then record the stream in another terminal::

    recorder --session "test" --lsl "NDS-RawData" --recording-time 10

Make sure to change the variable::

    LOCAL_DATA = True

and replace the variable with the path to your data::

    RAW_DATA_PATH = "the_path_to_your_recorded_raw_data.npz"

"""
# %%
# Environment setup
# ------------------
LOCAL_DATA = False

# %%
# Set data source
# ----------------
# Retrieve the data from AWS S3 or define the path to your local file.
from urllib.parse import urljoin

import pooch

DOWNLOAD_BASE_URL = "https://neural-data-simulator.s3.amazonaws.com/sample_data/v1/"

if not LOCAL_DATA:
    RAW_DATA_PATH = pooch.retrieve(
        url=urljoin(DOWNLOAD_BASE_URL, "example_NDS-RawData.npz"),
        known_hash="md5:887d88387674d8a7d27726e11663eee4",
    )
else:
    RAW_DATA_PATH = "the_path_to_your_recorded_raw_data.npz"

# %%
# Load data
# ---------
# Load the data to be plotted.
import numpy as np

raw_data_file = np.load(RAW_DATA_PATH)
raw_data = raw_data_file["data"] / 4
raw_data_timestamps = raw_data_file["timestamps"] - raw_data_file["timestamps"][0]


CHANNEL = 20
SAMPLING_RATE = 30000

# %%
# Estimate PSD using scipy
# -------------------------
import matplotlib.pyplot as plt
from scipy import signal

(f, S) = signal.welch(raw_data[:, CHANNEL], SAMPLING_RATE, nperseg=2**18)

plt.semilogy(f, S)
plt.xlim([0, 1000])
plt.xlabel("frequency [Hz]")
plt.ylabel("PSD [V**2/Hz]")
plt.show()

# %%
# Estimate PSD using NDS
# -----------------------
from neural_data_simulator.noise import calc_psd

(f, S) = calc_psd(raw_data[:, CHANNEL], SAMPLING_RATE)

plt.semilogy(f, S)
plt.xlim([0, 1000])
plt.xlabel("frequency [Hz]")
plt.ylabel("PSD [V**2/Hz]")
plt.show()
