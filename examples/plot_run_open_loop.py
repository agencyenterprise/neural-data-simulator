# flake8: noqa
"""
=========================================
Collect data to train your own decoder
=========================================

.. currentmodule:: neural_data_simulator

The goal of this example is to show how you can run an open loop simulation and collect the electrophysiology data
needed for training your own decoder.

By default, this script will download the data to be plotted from AWS S3. If you prefer to use
your own data, run the following steps:

1. Configure center out reach to `run without a decoder <../utilities_and_examples.html#running-without-a-decoder>`_.

2. Run the following script::

    # for macOS or Linux
    {center_out_reach; pkill -SIGTERM -f recorder} &; encoder &; ephys_generator &; \\
    recorder --session "open_loop" --lsl "NDS-RawData"; pkill -f decoder; \\
    pkill -f ephys_generator; pkill -f encoder; pkill -f center_out_reach

    # for Windows (PowerShell)
    Start-Process center_out_reach; Start-Process encoder; Start-Process ephys_generator; `
    recorder --session "example" --lsl "NDS-RawData"

3. Press CTRL+C to stop the recording or add a *\--recording_time n* to the recorder script to specify the amount of seconds to record

4. Change the following variable below::

    LOCAL_DATA = True

5. Replace the variable with the path to your data::

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


# %%
# Plot data
# ----------
import matplotlib.pyplot as plt

CHANNEL = 20

plt.plot(raw_data_timestamps, raw_data[:, CHANNEL])
plt.show()
