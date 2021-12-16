import librosa.display
import noisereduce as nr
import scipy.signal as sg
import soundfile as sf

from plots import *


def pre_processing(signal_data, file_name):
    # === Pre-Emphasis ===
    # Parameters:
    #   signal_data: A nparray with the original signal.
    #   file_name: A string that contains the file name.

    signal_emphasized = librosa.effects.preemphasis(signal_data)

    # === Filtering ===
    # Remove the background noise from the audio file.
    signal_reduced_noise = remove_noise(signal_data)

    # Remove the silent parts of the audio that are less than 40dB
    signal_filtered, _ = librosa.effects.trim(signal_reduced_noise, TOP_DB)

    signal_zcr = librosa.feature.zero_crossing_rate(signal_filtered)
    zcr_average = np.mean(signal_zcr)

    signal_short_time_energy = calculate_short_time_energy(signal_filtered)

    # Show plots
    show_plot_emphasized(signal_data, signal_emphasized)
    show_plots_compare_two_signals(signal_data, signal_reduced_noise)
    show_plot_zcr(signal_zcr)
    show_plot_short_time_energy(signal_filtered, signal_short_time_energy)

    # Exporting the filtered audio file.
    filtered_file_path = ".\\data\\samples\\" + file_name + "_filtered.wav"
    sf.write(filtered_file_path, signal_filtered, DEFAULT_SAMPLE_RATE)

    # Print statistics
    print(TXT_LINE, "\n")
    print(TXT_PRE_PROCESSING_STATISTICS)
    print(TXT_ORIGINAL_AUDIO_SAMPLE_RATE.format(DEFAULT_SAMPLE_RATE))
    print(TXT_AUDIO_ORIGINAL_DURATION_FORMAT.format(
        round(librosa.get_duration(signal_data, sr=DEFAULT_SAMPLE_RATE), 2))
    )
    print(TXT_AUDIO_FILTERED_DURATION_FORMAT.format(
        round(librosa.get_duration(signal_filtered, sr=DEFAULT_SAMPLE_RATE), 2))
    )
    print(TXT_ZCR_AVERAGE.format(zcr_average), "\n")
    print(TXT_LINE)

    return signal_filtered


def remove_noise(signal_data):
    # Parameters:
    #   signal_data: A nparray with the original signal.

    reduced_noise = nr.reduce_noise(audio_clip=signal_data,
                                    noise_clip=signal_data)

    return reduced_noise


def calculate_short_time_energy(signal_data):
    # Parameters:
    #   signal_data: A nparray with the original signal.

    signal = np.array(signal_data, dtype=float)
    win = sg.get_window("hamming", 301)

    if isinstance(win, str):
        win = sg.get_window(win, max(1, len(signal) // 8))
    win = win / len(win)

    signal_short_time_energy = sg.convolve(signal ** 2, win ** 2, mode="same")

    return signal_short_time_energy


def digits_segmentation(signal_nparray):
    # Parameters:
    #   signal_data: A nparray with the filtered signal.

    # We reverse the signal nparray.
    signal_reverse = signal_nparray[::-1]

    frames = librosa.onset.onset_detect(signal_nparray, sr=DEFAULT_SAMPLE_RATE, hop_length=FRAME_LENGTH)
    times = librosa.frames_to_time(frames, sr=DEFAULT_SAMPLE_RATE, hop_length=FRAME_LENGTH)
    samples = librosa.frames_to_samples(frames, FRAME_LENGTH)

    frames_reverse = librosa.onset.onset_detect(signal_reverse, sr=DEFAULT_SAMPLE_RATE, hop_length=FRAME_LENGTH)
    times_reverse = librosa.frames_to_time(frames_reverse, sr=DEFAULT_SAMPLE_RATE, hop_length=FRAME_LENGTH)

    for i in range(0, len(times_reverse) - 1):
        times_reverse[i] = WINDOW_LENGTH - times_reverse[i]
        i += 1

    times_reverse = sorted(times_reverse)

    i = 0
    while i < len(times_reverse) - 1:
        if times_reverse[i + 1] - times_reverse[i] < 1:
            times_reverse = np.delete(times_reverse, i)
            i -= 1
        i += 1

    i = 0
    while i < len(times) - 1:
        if times[i + 1] - times[i] < 1:
            times = np.delete(times, i + 1)
            frames = np.delete(frames, i + 1)
            samples = np.delete(samples, i + 1)
            i = i - 1
        i = i + 1

    merged_times = [*times, *times_reverse]
    merged_times = sorted(merged_times)

    samples = librosa.time_to_samples(merged_times, sr=DEFAULT_SAMPLE_RATE)

    return samples


def valid_digits(signal_data, samples):
    # Parameters:
    #   signal_data: An nparray with the signal.
    #   samples: An ndarray that contains integers.

    count_digits = 0
    digit = {}

    for i in range(0, len(samples), 2):
        if len(samples) % 2 == 1 and i == len(samples) - 1:
            digit[count_digits] = signal_data[samples[i - 1]:samples[i]]
        else:
            digit[count_digits] = signal_data[samples[i]:samples[i + 1]]
        count_digits += 1

    return digit


def recognition(digits, signal_data, dataset):
    # === Recognition of Digits ===
    # Parameters:
    #   digits: An array containing integer digits.
    #   signal_data: A nparray with the original signal for comparison.
    #   dataset: An array with all training signals.

    # Init an array that will contain our recognized digits in string.
    recognized_digits_array = []
    for digit in digits:
        cost_matrix_new = []
        mfccs = []

        mfcc_digit = librosa.feature.mfcc(y=digit,
                                          S=signal_data,
                                          sr=DEFAULT_SAMPLE_RATE,
                                          hop_length=FRAME_LENGTH,
                                          n_mfcc=13)
        mfcc_digit_mag = librosa.amplitude_to_db(abs(mfcc_digit))

        # 0-9 from training set
        for i in range(len(dataset)):
            # We basically filter the training dataset as well.
            dataset[i] = filter_dataset_signal(dataset[i].astype(np.float))

            # MFCC for each digit from the training set
            mfcc = librosa.feature.mfcc(y=dataset[i],
                                        S=signal_data,
                                        sr=DEFAULT_SAMPLE_RATE,
                                        hop_length=80,
                                        n_mfcc=13)

            # logarithm of the features ADDED
            mfcc_mag = librosa.amplitude_to_db(abs(mfcc))

            # apply dtw
            cost_matrix, wp = librosa.sequence.dtw(X=mfcc_digit_mag, Y=mfcc_mag)

            # make a list with minimum cost of each digit
            cost_matrix_new.append(cost_matrix[-1, -1])
            mfccs.append(mfcc_mag)

        # index of MINIMUM COST
        index_min_cost = cost_matrix_new.index(min(cost_matrix_new))

        recognized_digits_array.append(DATASET_SPLIT_LABELS[index_min_cost])

        for i in dataset:
            show_mel_spectrogram(dataset[i], DATASET_SPLIT_LABELS[index_min_cost])

    return recognized_digits_array


def get_training_samples_signal():
    # Initialize an array to append the signals of the training samples.
    training_samples_signals = {}

    index = 0
    # Loop between a range of 0-9, 0 in range(10) is 0 to 9 in python.
    for i in range(10):
        # Loop between the labels, s1 means sample1 and so on.
        for name in DATASET_SPLIT_LABELS:
            # Load the signal and add it to our array.
            training_samples_signals[index], _ = librosa.load(".\\data\\training\\"
                                                              + str(i)
                                                              + "_"
                                                              + name
                                                              + AUDIO_WAV_EXTENSION,
                                                              sr=DEFAULT_SAMPLE_RATE)

    index += 1

    return training_samples_signals


def filter_dataset_signal(signal_data):
    # === Filtering ===
    # Parameters:
    #   signal_data: A nparray with the signal.

    # Remove the background noise from the audio file.
    signal_reduced_noise = remove_noise(signal_data)

    # Remove the silent parts of the audio that are less than 40dB
    signal_filtered, _ = librosa.effects.trim(signal_reduced_noise, TOP_DB)

    return signal_filtered



