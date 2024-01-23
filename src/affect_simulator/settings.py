# Settings
eeg_ch_names = ["REF", "F3", "F4", "P3", "P4", "T7", "T8", "CZ"]
ch_types = "eeg"
trial_len = 60 # trial length in seconds
ecg_stream_name = 'BrainVision RDA'
marker_stream_name = 'psychopy_marker_oddball'
eeg_stream_name = 'g.USBamp'
downsample_sfreq = 128
markers_dict = {
    'start_practice_block': 0,
    'end_practice_block': 1,
    'start_baseline': 2,
    'end_baseline': 3,
    'start_exp_task': 4,
    'end_exp_task': 5,
    'cross': 6,
    'trial_start': 7,
    'trial_end': 8
    }