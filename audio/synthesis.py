import librosa
import numpy as np


def spectrogram_to_wav(mag, win_length, hop_length, n_fft, n_iter):
    """
    Convert a linear scale magnitude spectrogram into an audio time series.

    Arguments:
        mag (np.ndarray):
            Linear scale magnitude spectrogram.

        win_length (int):
            Length of each frame in audio samples.
            The window length is required to fulfill the condition `win_length` <= `n_fft`.

        hop_length (int):
            Number of audio samples to hop between frames.

        n_fft (int):
            FFT window size.

        n_iter (int):
            Number of reconstruction iterations used for the Griffin-Lim algorithm.

    Returns:
        (np.ndarray):
            Audio time series.
            The shape of the returned array is shape=(n,) and the arrays dtype is np.float32.

    Notes:
        - In future implementations we can drop Griffin-Lim based reconstruction.
        - Faster techniques or even implementations may be preferable.
    """
    wav = griffin_lim_v2(mag, win_length=win_length,
                         hop_length=hop_length,
                         n_fft=n_fft,
                         n_iter=n_iter)

    return wav.astype(np.float32)


def griffin_lim_v2(spectrogram, win_length, hop_length, n_fft, n_iter):
    """
    Applies Griffin-Lim reconstruction including simple phase estimation.

    Arguments:
        spectrogram (np.ndarray):
            Linear scale magnitude spectrogram.

        win_length (int):
            Length of each frame in audio samples.
            The window length is required to fulfill the condition `win_length` <= `n_fft`.

        hop_length (int):
            Number of audio samples to hop between frames.

        n_fft (int):
            FFT window size.

        n_iter (int):
            Number of reconstruction iterations to be used.

    Returns:
        (np.ndarray):
            Audio time series.
            The shape of the returned array is shape=(n,) and the arrays dtype is np.float32.

    Notes:
        This implementation is derived from:
            - https://github.com/librosa/librosa/issues/434
    """
    # Based on: https://github.com/librosa/librosa/issues/434

    # TODO: The code is extremely slow. We should try to implement a faster version.
    # TODO: Another approach to speed this up could be to implement it on the gpu using tf.signal.
    window = 'hann'

    # Initialize the phase component.
    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

    # Note: Instead of randomly initializing an estimated signal (just to immediately replace its
    # magnitude components during the first reconstruction iteration) we only randomly initialize
    # the phases.

    for i in range(n_iter):
        # Substitution of the estimated STFT magnitudes (STFTM) with the given target STFTM.
        full = np.abs(spectrogram).astype(np.complex) * angles

        # Revert the estimated STFT back into a time domain signal.
        estimated_signal = librosa.istft(full,
                                         n_fft=n_fft,
                                         win_length=win_length,
                                         hop_length=hop_length,
                                         window=window)

        # Compute the STFT from the estimated time domain signal.
        estimated_stft = librosa.stft(estimated_signal,
                                      n_fft=n_fft,
                                      win_length=win_length,
                                      hop_length=hop_length,
                                      window=window)

        # Extract the phase components from the estimated STFT.
        angles = np.exp(1j * np.angle(estimated_stft))

        # Reconstruction quality measurement for debugging purposes.
        # if False:
        #     diff = np.abs(spectrogram) - np.abs(estimated_stft)
        #     print("loss:", np.linalg.norm(diff, 'fro'))

    # Calculate the final estimate STFT.
    full = np.abs(spectrogram).astype(np.complex) * angles

    # Revert the estimated STFT back into a time domain signal.
    estimated_signal = librosa.istft(full,
                                     n_fft=n_fft,
                                     win_length=win_length,
                                     hop_length=hop_length,
                                     window=window)

    return estimated_signal
