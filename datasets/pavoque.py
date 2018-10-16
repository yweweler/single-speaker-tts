import os

import numpy as np

from audio.conversion import ms_to_samples, magnitude_to_decibel, normalize_decibel
from audio.effects import silence_interval_from_spectrogram
from audio.features import linear_scale_spectrogram, mel_scale_spectrogram
from audio.io import load_wav
from datasets.dataset_helper import DatasetHelper
from datasets.statistics import collect_duration_statistics
from tacotron.params.model import model_params


class PAVOQUEDatasetHelper(DatasetHelper):
    """
    Dataset loading helper for the PAVOQUE v0.2 dataset.
    """
    # TODO: Update the decibel values.
    # Mel. scale spectrogram reference dB over the entire dataset.
    mel_mag_ref_db = 12.63

    # Mel. scale spectrogram maximum dB over the entire dataset.
    mel_mag_max_db = 100.0

    # Linear scale spectrogram reference dB over the entire dataset.
    linear_ref_db = 24

    # Linear scale spectrogram maximum dB over the entire dataset.
    linear_mag_max_db = 100.0

    # Raw waveform silence reference signal dB.
    raw_silence_db = -15.0

    def __init__(self, dataset_folder, char_dict, fill_dict):
        super().__init__(dataset_folder, char_dict, fill_dict)

        self._abbreviations = {
            '(': '',
            ')': '',
            '[': '',
            ']': '',
            '-': '',
            'é': 'e',
            'ô': 'o',
            'ś': 's',
            'ê': 'e',
            'î': 'i',
            'š': 's',
            'í': 'i',
            'è': 'e',
            'à': 'a',
            'ć': 'c',
            'á': 'a',
            'ó': 'o',
            '´': ''
        }

    def load(self, max_samples=None, min_len=5, max_len=90, listing_file_name='neutral.txt'):
        data_file = os.path.join(self._dataset_folder, listing_file_name)
        # wav_folder = os.path.join(self._dataset_folder, 'wavs')

        file_paths = []
        sentences = []
        with open(data_file, 'r') as listing_file:
            # Iterate the file listing file.
            for line in listing_file:
                line = line.replace('\n', '')
                wav_path, normalized_sentence = line.split(' | ')

                # Extract the transcription.
                sentence = normalized_sentence

                # Skip sentences in case they do not meet the length requirements.
                sentence_len = len(sentence)
                if min_len is not None:
                    if sentence_len < min_len:
                        continue

                # Skip sentences in case they do not meet the length requirements.
                if max_len is not None:
                    if sentence_len > max_len:
                        continue

                sentences.append(sentence)

                # Get the audio file path.
                # TODO: '../' is a hack since the listing paths contain the base folder too.
                file_path = os.path.join(self._dataset_folder, '../', wav_path)
                file_paths.append(file_path)

                if max_samples is not None:
                    if len(sentences) == max_samples:
                        break

        # Normalize sentences, convert the characters to dictionary ids and determine their lengths.
        id_sentences, sentence_lengths = self.process_sentences(sentences)

        # for k, v in self._char2idx_dict.items():
        #     print("'{}': {},".format(k, v))

        return id_sentences, sentence_lengths, file_paths

    @staticmethod
    def load_audio(file_path):
        # Window length in audio samples.
        win_len = ms_to_samples(model_params.win_len, model_params.sampling_rate)
        # Window hop in audio samples.
        hop_len = ms_to_samples(model_params.win_hop, model_params.sampling_rate)

        # Load the actual audio file.
        wav, sr = load_wav(file_path.decode())

        # Calculate the linear scale spectrogram.
        # Note the spectrogram shape is transposed to be (T_spec, 1 + n_fft // 2) so dense layers
        # for example are applied to each frame automatically.
        linear_spec = linear_scale_spectrogram(wav, model_params.n_fft, hop_len, win_len).T

        # TODO: Experimental noise removal <64Hz
        linear_spec[:, 0:8] = 0

        # Convert the linear spectrogram into decibel representation.
        linear_mag = np.abs(linear_spec)
        linear_mag_db = magnitude_to_decibel(linear_mag)

        linear_mag_db = normalize_decibel(linear_mag_db,
                                          PAVOQUEDatasetHelper.linear_ref_db,
                                          PAVOQUEDatasetHelper.linear_mag_max_db)
        # => linear_mag_db.shape = (T_spec, 1 + n_fft // 2)

        # Calculate how many frames we have to crop at the beginning and end to remove silence.
        trim_start, trim_end = silence_interval_from_spectrogram(linear_mag_db,
                                                                 PAVOQUEDatasetHelper.raw_silence_db,
                                                                 np.max)

        # Calculate the Mel. scale spectrogram.
        # Note the spectrogram shape is transposed to be (T_spec, n_mels) so dense layers for
        # example are applied to each frame automatically.
        mel_spec = mel_scale_spectrogram(wav, model_params.n_fft, sr, model_params.n_mels,
                                         model_params.mel_fmin, model_params.mel_fmax, hop_len,
                                         win_len, 1).T

        # Convert the mel spectrogram into decibel representation.
        mel_mag = np.abs(mel_spec)
        mel_mag_db = magnitude_to_decibel(mel_mag)
        mel_mag_db = normalize_decibel(mel_mag_db,
                                       PAVOQUEDatasetHelper.mel_mag_ref_db,
                                       PAVOQUEDatasetHelper.mel_mag_max_db)
        # => mel_mag_db.shape = (T_spec, n_mels)

        # Remove silence at the beginning and end of the spectrogram's so the network does not have
        # to learn some random initial silence delay after which it is allowed to speak.
        linear_mag_db = linear_mag_db[trim_start:trim_end, :]
        mel_mag_db = mel_mag_db[trim_start:trim_end, :]

        # Tacotron reduction factor.
        if model_params.reduction > 1:
            mel_mag_db, linear_mag_db = DatasetHelper.apply_reduction_padding(mel_mag_db,
                                                                              linear_mag_db,
                                                                              model_params.reduction)

        return np.array(mel_mag_db).astype(np.float32), \
               np.array(linear_mag_db).astype(np.float32)


if __name__ == '__main__':
    init_char_dict = {
        'pad': 0,  # padding
        'eos': 1,  # end of sequence
        'i': 2, 'n': 3, ' ': 4, 's': 5, 'e': 6, 'r': 7, 'j': 8, 'u': 9, 'g': 10, 'd': 11, 'a': 12,
        'b': 13, 't': 14, 'c': 15, 'h': 16, 'l': 17, 'ä': 18, '.': 19, 'ü': 20, 'm': 21, 'p': 22,
        'w': 23, 'z': 24, ',': 25, 'ö': 26, 'o': 27, 'f': 28, 'k': 29, ';': 30, 'y': 31, 'v': 32,
        'x': 33, 'ß': 34, ':': 35, 'q': 36, '"': 37, '?': 38, '!': 39, "'": 40, '/': 41
    }

    dataset = PAVOQUEDatasetHelper(dataset_folder='/home/yves-noel/documents/master/thesis/datasets/PAVOQUE',
                                   char_dict=init_char_dict,
                                   fill_dict=False)

    ids, lens, paths = dataset.load()

    # Print a small sample from the dataset.
    # for p, s, l in zip(paths[:10], ids[:10], lens[:10]):
    #     print(p, np.fromstring(s, dtype=np.int32)[:10], l)

    # Collect and print the duration statistics for all the files.
    # collect_duration_statistics("PAVOQUE", paths)
