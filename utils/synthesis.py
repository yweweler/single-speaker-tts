import librosa
import numpy as np


def spectrogram_to_wav(mag, win_length, hop_length, n_fft, n_iter):
    wav = griffin_lim_v2(mag, win_length=win_length,
                         hop_length=hop_length,
                         n_fft=n_fft,
                         n_iter=n_iter)

    return wav.astype(np.float32)


def griffin_lim_v2(spectrogram, win_length, hop_length, n_fft, n_iter):
    # Based on: https://github.com/librosa/librosa/issues/434
    # TODO: This is rather slow. I really need to speed this up.
    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

    for i in range(n_iter):
        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(full, win_length=win_length, hop_length=hop_length, window="hann")
        rebuilt = librosa.stft(inverse, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hann")
        angles = np.exp(1j * np.angle(rebuilt))

        if False:
            diff = np.abs(spectrogram) - np.abs(rebuilt)
            print("loss:", np.linalg.norm(diff, 'fro'))

    full = np.abs(spectrogram).astype(np.complex) * angles
    inverse = librosa.istft(full, hop_length=hop_length, win_length=win_length, window="hann")

    return inverse
