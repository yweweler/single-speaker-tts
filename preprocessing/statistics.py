import numpy as np

from utils.audio import linear_scale_spectrogram, mel_scale_spectrogram
from utils.conversion import magnitude_to_decibel
from utils.io import load_wav


def decibel_statistics(wav, sampling_rate):
    # TODO: Add documentation.
    n_fft = 1024
    hop_length = n_fft // 4
    win_length = n_fft

    linear_spec = linear_scale_spectrogram(wav,
                                           n_fft=n_fft,
                                           hop_length=hop_length,
                                           win_length=win_length)

    mel_spec = mel_scale_spectrogram(wav,
                                     n_fft=n_fft,
                                     sampling_rate=sampling_rate,
                                     n_mels=80,
                                     fmin=0,
                                     fmax=8000,
                                     hop_length=hop_length,
                                     win_length=win_length,
                                     power=1)

    linear_mag = np.abs(linear_spec)
    linear_mag_db = magnitude_to_decibel(linear_mag)
    # linear_mag_db = normalize_decibel(linear_mag_db, 20, 100)

    mel_mag = np.abs(mel_spec)
    mel_mag_db = magnitude_to_decibel(mel_mag)
    # mel_mag_db = normalize_decibel(mel_mag_db, -7.7, 95.8)

    return np.array([np.min(linear_mag_db), np.max(linear_mag_db),
                     np.min(mel_mag_db), np.max(mel_mag_db)])


def collect_decibel_statistics(path_listing):
    # TODO: Add documentation.
    # min_linear_db
    # max_linear_db
    # min_mel_db
    # max_mel_db
    stats = np.zeros(4)

    for path in path_listing:
        wav, sampling_rate = load_wav(path)
        stats += decibel_statistics(wav, sampling_rate)

    n_files = len(path_listing)
    stats /= n_files

    return stats


if __name__ == '__main__':
    base_folder = '/tmp/TIMIT/'
    listing_file = 'train_all.txt'

    with open(base_folder + listing_file, 'r') as f:
        lines = f.readlines()

    wav_paths = [base_folder + line.split(',')[0] for line in lines]

    print("Collecting decibel statistics for {} files ...".format(len(wav_paths)))
    # Returns: (min_linear_db,  max_linear_db, min_mel_db, max_mel_db)
    min_linear_db, max_linear_db, min_mel_db, max_mel_db = collect_decibel_statistics(wav_paths)
    print("avg. min. linear magnitude (dB)", min_linear_db)
    print("avg. max. linear magnitude (dB)", max_linear_db)
    print("avg. max. mel magnitude (dB)", min_mel_db)
    print("avg. max. mel magnitude (dB)", max_mel_db)
