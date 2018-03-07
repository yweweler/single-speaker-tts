import copy
import math

import librosa
import numpy as np
import scipy
from scipy import signal


def spectrogram_to_wav(mag, win_length, hop_length, n_fft, n_iter):
    # TODO: Which one is the best?
    wav = griffin_lim_v1(mag, win_length=win_length,
                         hop_length=hop_length,
                         n_fft=n_fft,
                         n_iter=n_iter)

    wav = griffin_lim_v2(mag, win_length=win_length,
                         hop_length=hop_length,
                         n_fft=n_fft,
                         n_iter=n_iter)

    wav = griffin_lim_v3(mag, win_length=win_length,
                         hop_length=hop_length,
                         n_fft=n_fft,
                         n_iter=n_iter)

    return wav.astype(np.float32)


def griffin_lim_v1(spectrogram, win_length, hop_length, n_fft, n_iter):
    # Based on: https://github.com/Kyubyong/tacotron/blob/master/utils.py
    X_best = copy.deepcopy(spectrogram)

    for i in range(n_iter):
        X_t = librosa.istft(X_best, hop_length, win_length=win_length, window="hann")
        est = librosa.stft(X_t, n_fft, hop_length, win_length=win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase

        if True:
            diff = np.abs(spectrogram) - np.abs(est)
            print("loss:", np.linalg.norm(diff, 'fro'))

    X_t = librosa.istft(X_best, hop_length, win_length=win_length, window="hann")
    y = np.real(X_t)

    return y


def griffin_lim_v2(spectrogram, win_length, hop_length, n_fft, n_iter):
    # Based on: https://github.com/librosa/librosa/issues/434
    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

    for i in range(n_iter):
        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(full, win_length=win_length, hop_length=hop_length, window="hann")
        rebuilt = librosa.stft(inverse, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hann")
        angles = np.exp(1j * np.angle(rebuilt))

        if True:
            diff = np.abs(spectrogram) - np.abs(rebuilt)
            print("loss:", np.linalg.norm(diff, 'fro'))

    full = np.abs(spectrogram).astype(np.complex) * angles
    inverse = librosa.istft(full, hop_length=hop_length, win_length=win_length, window="hann")

    return inverse


def griffin_lim_v3(msgram, win_length, hop_length, n_fft, n_iter):
    # Based on: https://github.com/lonce/SPSI_Python/blob/master/spsi.py
    """
    Takes a 2D spectrogram ([freqs,frames]), the fft legnth (= widnow length) and the hope size (both in units of samples).
    Returns an audio signal.
    """

    numBins, numFrames = msgram.shape
    y_out = np.zeros(numFrames * hop_length + n_fft - hop_length)

    m_phase = np.zeros(numBins)
    m_win = scipy.signal.hanning(n_fft,
                                 sym=True)  # assumption here that hann was used to create the frames of the spectrogram

    # processes one frame of audio at a time
    for i in range(numFrames):
        m_mag = msgram[:, i]
        for j in range(1, numBins - 1):
            if m_mag[j] > m_mag[j - 1] and m_mag[j] > m_mag[j + 1]:  # if j is a peak
                alpha = m_mag[j - 1]
                beta = m_mag[j]
                gamma = m_mag[j + 1]
                denom = alpha - 2 * beta + gamma

                if denom != 0:
                    p = 0.5 * (alpha - gamma) / denom
                else:
                    p = 0

                # phaseRate=2*math.pi*(j-1+p)/fftsize    #adjusted phase rate
                phaseRate = 2 * math.pi * (j + p) / n_fft  # adjusted phase rate
                m_phase[j] = m_phase[j] + hop_length * phaseRate  # phase accumulator for this peak bin
                peakPhase = m_phase[j]

                # If actual peak is to the right of the bin freq
                if p > 0:
                    # First bin to right has pi shift
                    bin = j + 1
                    m_phase[bin] = peakPhase + math.pi

                    # Bins to left have shift of pi
                    bin = j - 1
                    while (bin > 1) and (m_mag[bin] < m_mag[bin + 1]):  # until you reach the trough
                        m_phase[bin] = peakPhase + math.pi
                        bin = bin - 1

                    # Bins to the right (beyond the first) have 0 shift
                    bin = j + 2
                    while (bin < numBins) and (m_mag[bin] < m_mag[bin - 1]):
                        m_phase[bin] = peakPhase
                        bin = bin + 1

                # if actual peak is to the left of the bin frequency
                if p < 0:
                    # First bin to left has pi shift
                    bin = j - 1
                    m_phase[bin] = peakPhase + math.pi

                    # and bins to the right of me - here I am stuck in the middle with you
                    bin = j + 1
                    while (bin < numBins) and (m_mag[bin] < m_mag[bin - 1]):
                        m_phase[bin] = peakPhase + math.pi
                        bin = bin + 1

                    # and further to the left have zero shift
                    bin = j - 2
                    while (bin > 1) and (m_mag[bin] < m_mag[bin + 1]):  # until trough
                        m_phase[bin] = peakPhase
                        bin = bin - 1

            # end ops for peaks
        # end loop over fft bins with

        magphase = m_mag * np.exp(1j * m_phase)  # reconstruct with new phase (elementwise mult)
        magphase[0] = 0
        magphase[numBins - 1] = 0  # remove dc and nyquist
        m_recon = np.concatenate([magphase, np.flip(np.conjugate(magphase[1:numBins - 1]), 0)])

        # overlap and add
        m_recon = np.real(np.fft.ifft(m_recon)) * m_win
        y_out[i * hop_length:i * hop_length + n_fft] += m_recon

    return y_out
