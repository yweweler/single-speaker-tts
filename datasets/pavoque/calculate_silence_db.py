import numpy as np

from audio.conversion import magnitude_to_decibel, ms_to_samples
from audio.effects import silence_interval_from_spectrogram
from audio.features import linear_scale_spectrogram
from audio.io import load_wav
from audio.visualization import plot_spectrogram
from datasets.statistics import collect_decibel_statistics
from tacotron.params.model import model_params


def plot_trim_silence(name, wav, sampling_rate, n_fft, hop_len, win_len):
    linear_spec = linear_scale_spectrogram(wav,
                                           n_fft=model_params.n_fft,
                                           hop_length=hop_len,
                                           win_length=win_len)
    linear_spec[0:8, ] = 0
    linear_spec_db = magnitude_to_decibel(np.abs(linear_spec))

    linear_spec_db = silence_interval_from_spectrogram(linear_spec_db, -15.0, np.max)
    if linear_spec_db is None:
        print('Warning: Silence trimming removed the whole waveform since all was silence.')
        return

    print(len(linear_spec.shape))

    plot_spectrogram(linear_spec_db,
                     sampling_rate=model_params.sampling_rate,
                     hop_length=hop_len,
                     fmin=model_params.mel_fmin,
                     fmax=model_params.mel_fmax,
                     y_axis='linear',
                     title='linear_spec_db: {}'.format(name))


if __name__ == '__main__':
    base_folder = 'datasets/pavoque/nosilence/'
    listing_file = 'nosilence.txt'

    # Read all lines from the TIMIT file listing.
    with open(base_folder + listing_file, 'r') as f:
        lines = f.readlines()

    # Extract only the wav file paths.
    wav_paths = [base_folder + line.replace('\n', '') for line in lines]

    for path in wav_paths:
        wav, sampling_rate = load_wav(path)
        win_len = ms_to_samples(model_params.win_len, sampling_rate=sampling_rate)
        hop_len = ms_to_samples(model_params.win_hop, sampling_rate=sampling_rate)

        plot_trim_silence(path, wav, sampling_rate, 2048, hop_len, win_len)

    exit()

    # Collect and print the decibel statistics for all the files.
    print("Collecting decibel statistics for {} files ...".format(len(wav_paths)))
    min_linear_db, max_linear_db, min_mel_db, max_mel_db = collect_decibel_statistics(wav_paths)
    print("avg. min. linear magnitude (dB)", min_linear_db)
    print("avg. max. linear magnitude (dB)", max_linear_db)
    print("avg. min. mel magnitude (dB)", min_mel_db)
    print("avg. max. mel magnitude (dB)", max_mel_db)
