import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence

def preprocess(eeg_data, channel_names, num_samples=1000, sampling_rate=200):
    # Assume eeg_data contains EEG signals
    eeg_signals = eeg_data[channel_names].values

    # Perform FFT for each channel
    fft_results = [np.fft.fft(eeg_signals[:, i]) for i in range(len(channel_names))]

    # Calculate frequencies
    frequencies = np.fft.fftfreq(len(fft_results[0]), 1/sampling_rate)

    top_frequencies = np.zeros((len(channel_names), 3))
    top_amplitudes = np.zeros((len(channel_names), 3))

    # Plot the FFT results for each channel
    for i, channel_name in enumerate(channel_names):
        plt.plot(frequencies, np.abs(fft_results[i]), label=channel_name)

        # Find the indices of the top 3 frequencies
        top_indices = np.argsort(np.abs(fft_results[i]))[-3:]

        # Extract the top 3 frequencies and their corresponding amplitudes
        top_frequencies[i] = frequencies[top_indices]
        top_amplitudes[i] = np.abs(fft_results[i][top_indices])

        # Print the top 3 frequencies and amplitudes for each channel

    # Calculate coherence between pairs of channels
    all_coherence_values = []
    for i in range(len(channel_names)-1):
        for j in range(i+1, len(channel_names)):
            # Compute coherence
            _, coherence_values = coherence(eeg_signals[:, i],
                                           eeg_signals[:, j],
                                           fs=sampling_rate, nperseg=256)
            all_coherence_values.extend(coherence_values)

    # Calculate average coherence across all channel pairs
    average_coherence = np.mean(all_coherence_values)
    return np.concatenate([top_frequencies[0], top_frequencies[1], top_frequencies[2], top_frequencies[3],
                           top_amplitudes[0], top_amplitudes[1], top_amplitudes[2], top_amplitudes[3], [average_coherence]])