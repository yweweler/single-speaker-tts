import abc
import csv
import os

import numpy as np


class DatasetHelper:
    def __init__(self, dataset_folder, char_dict, fill_dict):
        self._dataset_folder = dataset_folder
        self._char2idx_dict = char_dict
        self._fill_dict = fill_dict
        self._abbreviations = dict()

        # Initialize idx_dict in case a user defined dictionary was passed.
        self._idx2char_dict = dict()
        for char, _id in self._char2idx_dict.items():
            self._idx2char_dict[_id] = char

    def sent2idx(self, sentence):
        idx = [self._char2idx_dict[char] for char in sentence]
        return idx

    def idx2sent(self, idx):
        sentence = ''.join([self._idx2char_dict[char] for char in idx])
        return sentence

    def utf8_to_ascii(self, sentence):
        utf_sentence = bytes(sentence, 'utf-8')
        ascii_sentence = utf_sentence.decode('ascii', errors='ignore')
        return ascii_sentence

    def update_char_dict(self, sentence):
        # Update character dictionary with the chars contained in the sentence.
        for char in sentence:
            if char not in self._char2idx_dict.keys():
                # Character is not contained in the dictionary, we need to add it.
                _id = len(self._char2idx_dict.keys())
                self._char2idx_dict[char] = _id
                self._idx2char_dict[_id] = char

    def replace_abbreviations(self, sentence):
        for abbreviation, expansion in self._abbreviations.items():
            sentence = sentence.replace(abbreviation, expansion)

        return sentence

    def process_sentences(self, sentences):
        id_sequences = list()
        sequence_lengths = list()

        eos_token = self._char2idx_dict['eos']

        for sentence in sentences:
            # Make sentence lowercase.
            sentence = sentence.lower()

            # Replace abbreviations.
            sentence = self.replace_abbreviations(sentence)

            # Update character dictionary with the chars contained in the sentence.
            if self._fill_dict:
                self.update_char_dict(sentence)

            # Convert each character into its dictionary index.
            idx = self.sent2idx(sentence)

            # Append the EOS token to the sentence.
            idx.append(eos_token)

            # Append str. representation so that tf.Tensor handles this as an collection of objects.
            # This allows us to store sequences of different length in a single tensor.
            id_sequences.append(np.array(idx, dtype=np.int32).tostring())
            sequence_lengths.append(len(sentence) + 1)

        return id_sequences, sequence_lengths

    @abc.abstractmethod
    def load(self, max_samples):
        raise NotImplementedError

    @abc.abstractmethod
    def load_audio(self, file_path):
        raise NotImplementedError


class LJSpeechDatasetHelper(DatasetHelper):
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

        self._avg_min_lin_mag_db = -100.0
        self._avg_max_lin_mag_db = 35.7
        self._avg_min_mel_mag_db = -100
        self._avg_max_mel_mag_db = 6.0

    def load(self, max_samples=None):
        data_file = os.path.join(self._dataset_folder, 'metadata.csv')
        wav_folder = os.path.join(self._dataset_folder, 'wavs')

        file_paths = []
        sentences = []
        with open(data_file, 'r') as csv_file:
            csv_file_iter = csv.reader(csv_file, delimiter='|', quotechar='|')

            # Iterate the metadata file.
            for i, (file_id, _, normalized_sentence) in enumerate(csv_file_iter):
                # Get the audio file path.
                file_path = '{}.wav'.format(os.path.join(wav_folder, file_id))
                file_paths.append(file_path)

                # Extract the transcription.
                ascii_sentence = self.utf8_to_ascii(normalized_sentence)
                sentences.append(ascii_sentence)

                if max_samples is not None:
                    if (i+1) == max_samples:
                        break

        # Normalize sentences, convert the characters to dictionary ids and determine their lengths.
        id_sentences, sentence_lengths = self.process_sentences(sentences)

        # for k, v in self._char2idx_dict.items():
        #     print("'{}': {},".format(k, v))

        return id_sentences, sentence_lengths, file_paths

    def load_audio(self, file_path):
        raise NotImplementedError


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

    dataset = LJSpeechDatasetHelper(dataset_folder='/home/yves-noel/downloads/LJSpeech-1.1',
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


