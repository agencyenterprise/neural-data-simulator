import pyxdf
import mne
import numpy as np
from asrpy import ASR
import matplotlib.pyplot as plt

from settings import *

def read_xdf(file_path):
    # Read XDF
    streams, header = pyxdf.load_xdf(file_path)
    # Find stream indices
    # Customise according to your stream names
    ECG_index = -1
    EEG_index = -1
    MARKERS_index = -1
    for index in range(len(streams)):
        if streams[index]['info']['name'][0] == ecg_stream_name:
            ECG_index = index
        elif streams[index]['info']['name'][0] == marker_stream_name:
            MARKERS_index = index
        elif streams[index]['info']['name'][0] == eeg_stream_name:
            EEG_index = index

        if EEG_index >= 0 and ECG_index >= 0 and MARKERS_index >= 0:
            break

    sfreq = int(streams[EEG_index]['info']['nominal_srate'][0])
    # TODO: Consider using effective srate instead of nominal srate
    # sfreq = streams[EEG_index]['info']['effective_srate']

    # Preprocessing
    # 1. Create MNE object
    info = mne.create_info(ch_names=eeg_ch_names, sfreq=sfreq, ch_types=ch_types)
    EEG_channels = streams[EEG_index]["time_series"][:, :len(eeg_ch_names)].T
    raw = mne.io.RawArray(EEG_channels, info)
    raw = raw.drop_channels(['REF'])

    # 2. Remove powerline and low frequency noise
    notched = raw.notch_filter([50, 60])
    filtered_low_frequency = notched.filter(0.75, None, fir_design='firwin')

    # 3. Remove artifacts
    asr = ASR(sfreq=filtered_low_frequency.info["sfreq"], cutoff=13)
    asr.fit(filtered_low_frequency)
    artifacts_removed = asr.transform(filtered_low_frequency)

    # 4. Common-average referencing
    referenced = mne.set_eeg_reference(artifacts_removed, ref_channels='average')

    # 5. Band-pass filter
    bandpassed = referenced[0].filter(4, 45, fir_design='firwin')

    # 5. Downsample
    downsampled = bandpassed.copy().resample(sfreq=downsample_sfreq)
    downsampled_array = downsampled.get_data()

    # Get trials
    EEG_channels = downsampled_array.astype("float")
    # EEG_time = streams[EEG_index]["time_stamps"]
    EEG_time_zero_ref = streams[EEG_index]["time_stamps"] - streams[EEG_index]["time_stamps"][0]

    MARKER_stim = streams[MARKERS_index]["time_series"]
    # MARKER_time = streams[MARKERS_index]["time_stamps"]
    MARKER_time_zero_ref = streams[MARKERS_index]["time_stamps"] - streams[MARKERS_index]["time_stamps"][0]

    trials = []
    # Get index of relevant markers (trial start)
    markers_list = list(np.where(MARKER_stim == markers_dict['trial_start']))[0]
    for m in markers_list:
        # Trial start timestamp
        marker_start_time = MARKER_time_zero_ref[m]
        # Get the index of the first EEG sample recorded immediately after the marker was received
        EEG_start_idx = np.where(EEG_time_zero_ref >= marker_start_time)
        EEG_start_idx = EEG_start_idx[0][0]
        # Trial end index in the EEG stream
        EEG_end_idx = EEG_start_idx + (trial_len * sfreq)
        # Subset trial
        trial = np.array([channel[EEG_start_idx:EEG_end_idx] for channel in EEG_channels])
        # TODO: Handle difference between marker timestamp and EEG timestamp
        # Interpolate voltage between subsequent samples?
        # time_offset = EEG_start - marker_start_time
        trials.append(trial)

    return trials