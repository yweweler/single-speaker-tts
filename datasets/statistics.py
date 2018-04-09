import numpy as np

from audio.conversion import magnitude_to_decibel
from audio.features import linear_scale_spectrogram, mel_scale_spectrogram
from audio.io import load_wav


def decibel_statistics(wav, sampling_rate):
    """
    Calculate (min, max) values for the decibel values of both
    the linear scale magnitude spectrogram and a
    mel scale magnitude spectrogram.

    Arguments:
        wav (np.ndarray):
            Audio time series.
            The shape is expected to be shape=(n,).

        sampling_rate (int):
            Sampling rate using in the calculation of `wav`.

    Returns:
        np.ndarray:
            Min and max values of the decibel representations.

            Calculation: np.array[min(linear_db), max(linear_db), min(mel_db), max(mel_db)]
    """
    n_fft = 1024
    hop_length = n_fft // 4
    win_length = n_fft
    n_mels = 80

    # Get the linear scale spectrogram.
    linear_spec = linear_scale_spectrogram(wav,
                                           n_fft=n_fft,
                                           hop_length=hop_length,
                                           win_length=win_length)

    # Get the mel scale spectrogram.
    mel_spec = mel_scale_spectrogram(wav,
                                     n_fft=n_fft,
                                     sampling_rate=sampling_rate,
                                     n_mels=n_mels,
                                     fmin=0,
                                     fmax=sampling_rate // 2,
                                     hop_length=hop_length,
                                     win_length=win_length,
                                     power=1)

    # Convert the linear spectrogram into decibel representation.
    linear_mag = np.abs(linear_spec)
    linear_mag_db = magnitude_to_decibel(linear_mag)
    # linear_mag_db = normalize_decibel(linear_mag_db, 20, 100)

    # Convert the mel spectrogram into decibel representation.
    mel_mag = np.abs(mel_spec)
    mel_mag_db = magnitude_to_decibel(mel_mag)
    # mel_mag_db = normalize_decibel(mel_mag_db, -7.7, 95.8)

    return np.array([
        np.min(linear_mag_db), np.max(linear_mag_db),
        np.min(mel_mag_db), np.max(mel_mag_db)
    ])


def collect_decibel_statistics(path_listing):
    """
    Calculate the average (min, max) values for the decibel values of
    both the linear scale magnitude spectrogram's and a mel scale
    magnitude spectrogram's of a list of wav files.

    Arguments:
        path_listing (list):
            List of wav file paths.

    Returns:
        np.ndarray:
            Average min and max values of the decibel representations.

            Calculation: (avg(linear_min_db), avg(linear_max_db), avg(mel_min_db), avg(mel_max_db)).
    """
    # (min_linear, max_linear, min_mel, max_mel)
    stats = np.zeros(4)

    # Accumulate statistics for a list of wav files.
    for path in path_listing:
        wav, sampling_rate = load_wav(path)
        # Accumulate the calculated min and max values.
        stats += decibel_statistics(wav, sampling_rate)

    # Calculate the average min and max values.
    n_files = len(path_listing)
    stats /= n_files

    return stats


if __name__ == '__main__':
    base_folder = '/tmp/TIMIT/'
    listing_file = 'train_all.txt'

    # Read all lines from the TIMIT file listing.
    with open(base_folder + listing_file, 'r') as f:
        lines = f.readlines()

    # Extract only the wav file paths.
    wav_paths = [base_folder + line.split(',')[0] for line in lines]

    # Collect and print the decibel statistics for all the files.
    print("Collecting decibel statistics for {} files ...".format(len(wav_paths)))
    min_linear_db, max_linear_db, min_mel_db, max_mel_db = collect_decibel_statistics(wav_paths)
    print("avg. min. linear magnitude (dB)", min_linear_db)     # -99.99 dB
    print("avg. max. linear magnitude (dB)", max_linear_db)     # +20.08 dB
    print("avg. min. mel magnitude (dB)", min_mel_db)           # -95.73 dB
    print("avg. max. mel magnitude (dB)", max_mel_db)           # -07.74 dB
