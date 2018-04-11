import librosa
import numpy as np
import pysptk


# https://github.com/keithito/tacotron/blob/master/util/audio.py
# https://github.com/eYSIP-2017/eYSIP-2017_Speech_Spoofing_and_Verification


def mel_scale_spectrogram(wav, n_fft, sampling_rate, n_mels, fmin, fmax, hop_length, win_length,
                          power):
    """
    Calculate a Mel-scaled spectrogram from a signal.

    Arguments:
        wav (np.ndarray):
            Audio time series.
            The shape is expected to be shape=(n,).

        n_fft (int):
            FFT window size.

        sampling_rate (int):
            Sampling rate using in the calculation of `wav`.

        n_mels (int):
            Number of Mel bands to generate.

        fmin (float):
            Lowest frequency (in Hz).

        fmax (float):
            Highest frequency (in Hz).
            If `None`, use `fmax = sampling_rate / 2.0`.

        hop_length (int):
            Number of audio samples to hop between frames.

        win_length (int):
            Length of each frame in audio samples.
            The window length is required to fulfill the condition `win_length` <= `n_fft`.

        power (float):
            Exponent for the magnitudes of the linear-scale spectrogram.
            e.g., 1 for energy, 2 for power, etc.

    Returns:
        np.ndarray:
            STFT matrix of the Mel-scaled spectrogram.
            The shape of the matrix will be shape(n_mels, t)

    Notes:
        - Setting `win_length` to `n_fft` results in exactly the same values the librosa build in
          functions `librosa.feature.mfcc` and `librosa.feature.melspectrogram` would deliver.
        - This implementation is directly derived from the function
          `librosa.feature.melspectrogram`.
    """
    # This implementation calculates the Mel-scaled spectrogram and the mfccs step by step.
    # Both `librosa.feature.mfcc` and `librosa.feature.melspectrogram` could be used to do this in
    # fewer lines of code. However, they do not allow to control the window length used in the
    # initial stft calculation.
    # Setting `win_length` to `n_fft` results in exactly the same values the librosa build in
    # functions would deliver.

    # Short-time Fourier transform of the signal to create a linear-scale spectrogram.
    # Return shape: (n_fft/2 + 1, n_frames), with n_frames being floor(len(wav) / win_hop).
    mag_phase_spec = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    # Extract the magnitudes from the linear-scale spectrogram.
    # Return shape: (n_fft/2 + 1, n_frames).
    mag_spec = np.abs(mag_phase_spec)

    # Raise the linear-scale spectrogram magnitudes to the power of `power`.
    # `power` = 1 for energy,`power` = 2 for power, etc.
    # Return shape: (n_fft/2 + 1, n_frames).
    linear_spec = mag_spec ** power

    # Create a filter-bank matrix to combine FFT bins into Mel-frequency bins.
    # Return shape: (n_mels, n_fft/2 + 1).
    mel_basis = librosa.filters.mel(sr=sampling_rate,
                                    n_fft=n_fft,
                                    n_mels=n_mels,
                                    fmin=fmin,
                                    fmax=fmax,
                                    htk=True)

    # Apply Mel-filters to create a Mel-scaled spectrogram.
    # Return shape: (n_mels, n_frames).
    mel_spec = np.dot(mel_basis, linear_spec)

    return mel_spec


def calculate_mfccs(mel_spec, sampling_rate, n_mfcc):
    """
    Calculate Mel-frequency cepstral coefficients (MFCCs) from a Mel-scaled spectrogram.

    Arguments:
        mel_spec (np.ndarray):
            STFT matrix of the Mel-scaled spectrogram.
             The shape of the matrix is expected to be shape=(n_mels, t).

        sampling_rate (int):
            Sampling rate using in the calculation of `mel_spec`.

        n_mfcc (int):
            Number of MFCCs to return.

    Returns:
        np.ndarray:
            Mel-frequency cepstral coefficients (MFCCs) for each frame of the mel spectrogram.
            The shape of the array will be shape=(n_mfcc, t).
    """
    # Calculate Mel-frequency cepstral coefficients (MFCCs) from the Mel-spectrogram.
    # Return shape: (n_mfcc, n_frames).
    mfccs = librosa.feature.mfcc(S=mel_spec, sr=sampling_rate, n_mfcc=n_mfcc)

    return mfccs


def calculate_mceps(wav, n_fft, hop_length, n_mceps, alpha):
    """
    Mel-cepstrum analysis.

    Arguments:
        wav (np.ndarray):
            Audio time series.
            The shape is expected to be shape=(n,).

        n_fft (int):
            FFT window size.

        hop_length (int):
            Number of audio samples to hop between frames.

        n_mceps (int):
            Order of mel-cepstrum.

        alpha (float):
            All pass constant.

    Returns:
        np.ndarray: Mel-cepstrum.
    """
    win_length = n_fft
    frames = librosa.util.frame(y=wav,
                                frame_length=win_length,
                                hop_length=hop_length).astype(np.float64).T

    frames *= pysptk.blackman(win_length)

    mc = pysptk.mcep(frames, n_mceps, alpha)
    log_h = pysptk.mgc2sp(mc, alpha, 0.0, win_length).real

    return log_h.T, mc


def linear_scale_spectrogram(wav, n_fft, hop_length=None, win_length=None):
    """
    Short-time Fourier transform (STFT).

    Get a linear scale magnitude, phase spectrogram from an audio time series.

    Arguments:
        wav (np.ndarray):
            Audio time series.
            The shape is expected to be shape=(n,).

        n_fft (int):
            FFT window size.

        hop_length (int):
            Number of audio samples to hop between frames.

        win_length (int):
            Length of each frame in audio samples.
            The window length is required to fulfill the condition `win_length` <= `n_fft`.

    Returns:
        np.ndarray:
            Returns a complex-valued matrix D such that:
                - `np.abs(D[f, t])` is the magnitude of frequency bin `f` at frame `t`.
                - `np.angle(D[f, t])` is the phase of frequency bin `f` at frame `t`.

            The shape is of the matrix will be shape=(1 + n_fft/2, t).
    """
    return librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
