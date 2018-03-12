import numpy as np


def magnitude_to_decibel(mag):
    """
    Convert a magnitude spectrogram to decibel (dB).
    Decibel values will be in the range [-100 dB, +inf dB]

    :param mag: np.ndarray
         Magnitude spectrum.

    :return: np.ndarray
        Magnitude spectrum in decibel representation.

        20 * log10(max(1e-5, abs(mag)))
    """
    # Decibel values are bounded to a lower level of -100 dB.
    # 20 * log10(1e-5) = -100 dB
    #
    # There is no upper bound for the decibel values.
    # The 0 dB level is reached in case of magnitudes with value 1.0.
    # 20 * log10( 1.0) = 0 dB

    return 20.0 * np.log10(np.maximum(1e-5, mag))


def decibel_to_magnitude(mag_db):
    """
    Convert a magnitude spectrogram in decibel (dB) representation back to a raw magnitude spectrogram.

    :param mag_db: np.ndarray
        Magnitude spectrum in decibel representation.

    :return:
        Magnitude spectrum.

        power(10, mag_db / 20)
    """
    # TODO: Limit passed mag_db to [-100 dB, +inf dB] before conversion?
    mag = np.power(10.0, mag_db / 20.0)

    return mag


def normalize_decibel(db, ref_db, max_db):
    """
    Normalize decibel (dB) values to the range [0.0, 1.0].

    :param db:
        Data in decibel representation (dB).

    :param ref_db:
        Signal reference in decibel (dB).

    :param max_db:
        Signal maximum in decibel (dB).

    :return:
        clip((mag_db - ref_db + max_db) / max_db, 0.0, 1.0)
    """
    return np.clip((db - ref_db + max_db) / max_db, 0.0, 1.0)


def inv_normalize_decibel(norm_db, ref_db, max_db):
    """
    Convert normalized decibel (dB) values from the range [0.0, 1.0] back to decibel.

    :param norm_db:
        Normalized decibel values in the range [0.0, 1.0].

    :param ref_db:
        Signal reference in decibel (dB) used for normalization.

    :param max_db:
        Signal maximum in decibel (dB) used for normalization.

    :return:
        (clip(norm_db, 0.0, 1.0) * max_db) + ref_db - max_db
    """
    return (np.clip(norm_db, 0.0, 1.0) * max_db) + ref_db - max_db


def samples_to_ms(samples, sampling_rate):
    """
    Convert a duration in samples into milliseconds.

    :param samples: int > 0 [scalar]
        Samples to convert into milliseconds.

    :param sampling_rate: int > 0 [scalar]
        Sampling rate of `wav`.

    :return: float > 0 [scalar]
        Duration in ms.
    """
    return (samples / sampling_rate) * 1000


def ms_to_samples(ms, sampling_rate):
    """
    Convert a duration in milliseconds into samples.

    :param ms: float > 0 [scalar]
        Duration in ms.

    :param sampling_rate: int > 0 [scalar]
        Sampling rate of `wav`.

    :return: int > 0 [scalar]
        Duration in samples.
    """
    return int((ms / 1000) * sampling_rate)