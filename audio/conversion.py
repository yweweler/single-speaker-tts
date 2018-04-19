import numpy as np


def magnitude_to_decibel(mag):
    """
    Convert a magnitude spectrogram to decibel (dB).

    Decibel values will be in the range [-100 dB, +inf dB]

    Arguments:
        mag (np.ndarray):
            Magnitude spectrum.

    Returns:
        np.ndarray:
            Magnitude spectrum in decibel representation.

            Calculation: 20 * log10(max(1e-5, abs(mag)))

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
    Convert a magnitude spectrogram in decibel (dB) representation back to raw magnitude
    representation.

    Arguments:
        mag_db (np.ndarray):
            Magnitude spectrum in decibel representation.

    Returns:
        np.ndarray:
            Magnitude spectrum.

            Calculation: power(10, mag_db / 20)
    """
    if (mag_db < -100.0).any():
        raise AssertionError('"conversion.decibel_to_magnitude" was asked to convert a dB value '
                             'smaller -100 dB.')

    mag = np.power(10.0, mag_db / 20.0)

    return mag


def normalize_decibel(db, ref_db, max_db):
    """
    Normalize decibel (dB) values and map them to the range [0.0, 1.0].

    Arguments:
        db (np.ndarray):
            Data in decibel representation (dB).

        ref_db (float):
            Signal reference in decibel (dB).

        max_db (float):
            Signal maximum in decibel (dB).

    Returns:
        np.ndarray:
            Normalized values in the range [0.0, 1.0].

            Calculation: clip((db - ref_db + max_db) / max_db, 0.0, 1.0)
    """
    # np.clip(1.0 - (db - max_db) / (ref_db - max_db), 0.0, 1.0)
    return np.clip((db - ref_db + max_db) / max_db, 1e-8, 1.0)


def inv_normalize_decibel(norm_db, ref_db, max_db):
    """
    Convert normalized decibel (dB) values from the range [0.0, 1.0] back to decibel.

    Arguments:
        norm_db (np.ndarray):
            Normalized decibel values in the range [0.0, 1.0].

        ref_db (float):
            Signal reference in decibel (dB) used during normalization.

        max_db (float):
            Signal maximum in decibel (dB) used during normalization.

    Returns:
        np.ndarray:
            Input data converted to decibel representation.

            Calculation: (clip(norm_db, 0.0, 1.0) * max_db) + ref_db - max_db
    """
    return (np.clip(norm_db, 0.0, 1.0) * max_db) + ref_db - max_db


def samples_to_ms(samples, sampling_rate):
    """
    Convert a duration in samples into milliseconds.

    Arguments:
        samples (int):
            Samples to convert into milliseconds.

        sampling_rate (int):
            Sampling rate of of the signal.

    Returns:
        float: Duration in ms.
    """
    return (samples / sampling_rate) * 1000


def ms_to_samples(ms, sampling_rate):
    """
    Convert a duration in milliseconds into samples.

    Arguments:
        ms (float):
            Duration in ms.

        sampling_rate (int):
            Sampling rate of of the signal.

    Returns:
        int: Duration in samples.
    """
    return int((ms / 1000) * sampling_rate)
