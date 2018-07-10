import csv
import os

import librosa
import numpy as np

from audio.conversion import ms_to_samples, magnitude_to_decibel, normalize_decibel
from audio.features import linear_scale_spectrogram, mel_scale_spectrogram
from audio.io import load_wav
from datasets.dataset_helper import DatasetHelper
from tacotron.params.model import model_params


class LJSpeechDatasetHelper(DatasetHelper):
    """
    Dataset loading helper for the LJSpeech v1.1 dataset.
    """
    # Mel. scale spectrogram reference dB over the entire dataset.
    mel_mag_ref_db = 20

    # Mel. scale spectrogram maximum dB over the entire dataset.
    mel_mag_max_db = 100.0

    # Linear scale spectrogram reference dB over the entire dataset.
    linear_ref_db = 20

    # Linear scale spectrogram maximum dB over the entire dataset.
    linear_mag_max_db = 100.0

    # Raw waveform silence reference signal dB.
    raw_silence_db = None

    def __init__(self, dataset_folder, char_dict, fill_dict):
        super().__init__(dataset_folder, char_dict, fill_dict)

        self._abbreviations = {
            'mr.': 'mister',
            'mrs.': 'misses',
            'dr.': 'doctor',
            'no.': 'number',
            'st.': 'saint',
            'co.': 'company',
            'jr.': 'junior',
            'maj.': 'major',
            'gen.': 'general',
            'drs.': 'doctors',
            'rev.': 'reverend',
            'lt.': 'lieutenant',
            'hon.': 'honorable',
            'sgt.': 'sergeant',
            'capt.': 'captain',
            'esq.': 'esquire',
            'ltd.': 'limited',
            'col.': 'colonel',
            'ft.': 'fort',
            ':': '',
            ';': '',
            '(': '',
            ')': '',
            '[': '',
            ']': '',
            '-': '',
            ',': '',
            '.': '',
            '"': '',
            '\'': '',
            '!': '',
            '?': '',
        }

    def load(self, max_samples=None, min_len=30, max_len=90, listing_file_name='metadata.csv'):
        data_file = os.path.join(self._dataset_folder, listing_file_name)
        wav_folder = os.path.join(self._dataset_folder, 'wavs')

        file_paths = []
        sentences = []
        with open(data_file, 'r') as csv_file:
            csv_file_iter = csv.reader(csv_file, delimiter='|', quotechar='|')

            # Iterate the metadata file.
            for file_id, _, normalized_sentence in csv_file_iter:
                # Extract the transcription.
                # We do not want the sentence to contain any non ascii characters.
                sentence = self.utf8_to_ascii(normalized_sentence)

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
                file_path = '{}.wav'.format(os.path.join(wav_folder, file_id))
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

        # TODO: Determine a better silence reference level for the LJSpeech dataset (See: #9).
        # Remove silence at the beginning and end of the wav so the network does not have to learn
        # some random initial silence delay after which it is allowed to speak.
        wav, _ = librosa.effects.trim(wav)

        # Calculate the linear scale spectrogram.
        # Note the spectrogram shape is transposed to be (T_spec, 1 + n_fft // 2) so dense layers
        # for example are applied to each frame automatically.
        linear_spec = linear_scale_spectrogram(wav, model_params.n_fft, hop_len, win_len).T

        # Calculate the Mel. scale spectrogram.
        # Note the spectrogram shape is transposed to be (T_spec, n_mels) so dense layers for
        # example are applied to each frame automatically.
        mel_spec = mel_scale_spectrogram(wav, model_params.n_fft, sr, model_params.n_mels,
                                         model_params.mel_fmin, model_params.mel_fmax, hop_len,
                                         win_len, 1).T

        # Convert the linear spectrogram into decibel representation.
        linear_mag = np.abs(linear_spec)
        linear_mag_db = magnitude_to_decibel(linear_mag)
        linear_mag_db = normalize_decibel(linear_mag_db,
                                          LJSpeechDatasetHelper.linear_ref_db,
                                          LJSpeechDatasetHelper.linear_mag_max_db)
        # => linear_mag_db.shape = (T_spec, 1 + n_fft // 2)

        # Convert the mel spectrogram into decibel representation.
        mel_mag = np.abs(mel_spec)
        mel_mag_db = magnitude_to_decibel(mel_mag)
        mel_mag_db = normalize_decibel(mel_mag_db,
                                       LJSpeechDatasetHelper.mel_mag_ref_db,
                                       LJSpeechDatasetHelper.mel_mag_max_db)
        # => mel_mag_db.shape = (T_spec, n_mels)

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
        'p': 2,
        'r': 3,
        'i': 4,
        'n': 5,
        't': 6,
        'g': 7,
        ' ': 8,
        'h': 9,
        'e': 10,
        'o': 11,
        'l': 12,
        'y': 13,
        's': 14,
        'w': 15,
        'c': 16,
        'a': 17,
        'd': 18,
        'f': 19,
        'm': 20,
        'x': 21,
        'b': 22,
        'v': 23,
        'u': 24,
        'k': 25,
        'j': 26,
        'z': 27,
        'q': 28,
    }

    dataset = LJSpeechDatasetHelper(dataset_folder='/home/yves-noel/documents/master/thesis/datasets/LJSpeech-1.1',
                                    char_dict=init_char_dict,
                                    fill_dict=False)

    ids, lens, paths = dataset.load()

    # Print a small sample from the dataset.
    # for p, s, l in zip(paths[:10], ids[:10], lens[:10]):
    #     print(p, np.fromstring(s, dtype=np.int32)[:10], l)

    # Collect and print the decibel statistics for all the files.
    # print("Collecting decibel statistics for {} files ...".format(len(paths)))
    # min_linear_db, max_linear_db, min_mel_db, max_mel_db = collect_decibel_statistics(paths)
    # print("avg. min. linear magnitude (dB)", min_linear_db)
    # print("avg. max. linear magnitude (dB)", max_linear_db)
    # print("avg. min. mel magnitude (dB)", min_mel_db)
    # print("avg. max. mel magnitude (dB)", max_mel_db)
