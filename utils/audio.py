import librosa
import numpy as np
import pysptk

# https://github.com/keithito/tacotron/blob/master/util/audio.py

# https://github.com/eYSIP-2017/eYSIP-2017_Speech_Spoofing_and_Verification

AUDIO_FLOAT_EPS = 1e-7


def mel_scale_spectrogram(wav, n_fft, sampling_rate, n_mels, fmin, fmax, hop_length, win_length, power):
    """
    Calculate a Mel-scaled spectrogram from a signal.

    :param wav: np.ndarray [shape=(n,)]
        Audio time series.

    :param n_fft: int > 0 [scalar]
        FFT window size.

    :param sampling_rate: int > 0 [scalar]
        Sampling rate of `wav`.

    :param n_mels: int > 0 [scalar]
        Number of Mel bands to generate.

    :param fmin: float >= 0 [scalar]
        Lowest frequency (in Hz).

    :param fmax: float >= 0 [scalar]
        Highest frequency (in Hz).
        If `None`, use `fmax = sampling_rate / 2.0`.

    :param hop_length: int > 0 [scalar]
        Number of audio samples to hop between frames.

    :param win_length: int <= n_fft [scalar]
        Length of each frame in audio samples.

    :param power: float > 0 [scalar]
        Exponent for the magnitudes of the linear-scale spectrogram.
        e.g., 1 for energy, 2 for power, etc.

    :return:
        mel_spec: np.ndarray [shape=(n_mels, t)]
            STFT matrix of the Mel-scaled spectrogram.

    :notes:
        Setting `win_length` to `n_fft` results in exactly the same values the librosa build in functions
        `librosa.feature.mfcc` and `librosa.feature.melspectrogram` would deliver.
    """
    # This implementation calculates the Mel-scaled spectrogram and the mfccs step by step.
    # Both `librosa.feature.mfcc` and `librosa.feature.melspectrogram` could be used to do this in fewer lines of code.
    # However, they do not allow to control the window length used in the initial stft calculation.
    # Setting `win_length` to `n_fft` results in exactly the same values the librosa build in functions would deliver.

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
    mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=True)

    # Apply Mel-filters to create a Mel-scaled spectrogram.
    # Return shape: (n_mels, n_frames).
    mel_spec = np.dot(mel_basis, linear_spec)

    return mel_spec


def calculate_mfccs(mel_spec, sampling_rate, n_mfcc):
    """
    Calculate Mel-frequency cepstral coefficients (MFCCs) from a Mel-scaled spectrogram.

    :param mel_spec: np.ndarray [shape=(n_mels, t)]
        STFT matrix of the Mel-scaled spectrogram.

    :param sampling_rate: int > 0 [scalar]
        Sampling rate of `wav`.

    :param n_mfcc: int > 0 [scalar]
        Number of MFCCs to return.

    :return:
        mfccs: np.ndarray [shape=(n_mfcc, t)]
            Mel-frequency cepstral coefficients (MFCCs) for each frame.
    """
    # Calculate Mel-frequency cepstral coefficients (MFCCs) from the Mel-spectrogram.
    # Return shape: (n_mfcc, n_frames).
    mfccs = librosa.feature.mfcc(S=mel_spec, sr=sampling_rate, n_mfcc=n_mfcc)

    return mfccs


def calculate_mceps(wav, hop_len, n_mceps, alpha, n_fft):
    win_len = n_fft
    frames = librosa.util.frame(wav, frame_length=win_len,
                                hop_length=hop_len).astype(np.float64).T
    frames *= pysptk.blackman(win_len)

    mc = pysptk.mcep(frames, n_mceps, alpha)
    logH = pysptk.mgc2sp(mc, alpha, 0.0, win_len).real

    # print("logH mceps shape:", logH.T.shape)
    # print("mc mceps shape:", mc.T.shape)
    return logH.T, mc


def linear_scale_spectrogram(wav, n_fft, hop_length=None, win_length=None):
    linear_spec = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    # mag, phase = librosa.magphase(linear_spec)

    return linear_spec
