from termcolor import colored


# region Lang
# Console
TXT_INPUT_FILE = "Enter sound file path:\n"
TXT_FILE_NOT_FOUND = colored("File not found at:\n", "red")
TXT_FILE_WRONG_EXTENSION = colored("Wrong file extension. Please try again!", "red")
TXT_PRE_PROCESSING_STATISTICS = "PRE-PROCESSING STATISTICS:"
TXT_AUDIO_ORIGINAL_DURATION_FORMAT = "- Original Audio Duration: {} sec."
TXT_AUDIO_FILTERED_DURATION_FORMAT = "- Audio (Filtered) Duration: {} sec."
TXT_ORIGINAL_AUDIO_SAMPLE_RATE = "- Sample Rate: {}"
TXT_ZCR_AVERAGE = "- Average ZCR: {}"
TXT_DIGITS_FOUND = "[!] Total Digits Found: {}"
TXT_DIGITS_RECOGNIZED = "Digits Recognized:"
TXT_LINE = "==============================================="
# Plot
TXT_AMPLITUDE = "Amplitude"
TXT_TIME = "Time (s)"
TXT_FREQUENCY = "Frequency (Hz)"
TXT_ORIGINAL_SIGNAL = "Original Signal"
TXT_PRE_EMPHASIZED_SIGNAL = "Pre-Emphasized Signal"
TXT_DECIBELS = "Decibels (dB)"
TXT_MEL = "Mel Scale (Mel)"
TXT_ORIGINAL = "Original"
TXT_FILTERED = "Filtered"
TXT_STE = "STE"
TXT_ZERO_CROSSING_RATE = "Zero-Crossing Rate"
TXT_SHORT_TIME_ENERGY = "Short-Time Energy"
# endregion


# region Variables
DIRECTORY_PLOTS = ".\\data\\plots"
# Remove signal part if dB is less than 40
TOP_DB = 40
DEFAULT_SAMPLE_RATE = 16000
AUDIO_WAV_EXTENSION = ".wav"
# window length in sec. Default is 0.03.
WINDOW_LENGTH = 0.03
# step between successive windows in sec. Default is 0.01.
WINDOW_HOP = 0.01
FRAME_LENGTH = round(WINDOW_LENGTH * DEFAULT_SAMPLE_RATE)
DATASET_SPLIT_LABELS = ["s1", "s2", "s3"]
# endregion
