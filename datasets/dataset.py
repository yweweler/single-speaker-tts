"""
Defines and implements a basic dataset.
"""

import csv
import json
import os

from audio.io import load_wav
from datasets.utils import statistics


class Dataset:
    """
    Basic dataset that can be loaded and saved.

    The dataset consists of a dataset definition file `.json` and separate `.csv` files
    containing listings for the training and evaluation portions.
    """

    def __init__(self, dataset_file):
        self.__dataset_file = dataset_file
        self.__definition = dict()

    def get_definition(self):
        return self.__definition

    def get_train_listing_generator(self):
        train_listing_file = os.path.join(
            self.get_definition()['dataset_folder'],
            self.get_definition()['train_listing']
        )
        raw_rows = self.load_listing_file(train_listing_file)
        print('Loaded {} train rows'.format(len(raw_rows)))

        for row in raw_rows:
            parsed_row = self.parse_listing_row(row)
            yield parsed_row

    def get_eval_listing_generator(self):
        eval_listing_file = os.path.join(
            self.get_definition()['dataset_folder'],
            self.get_definition()['eval_listing']
        )
        raw_rows = self.load_listing_file(eval_listing_file)
        print('Loaded {} eval rows'.format(len(raw_rows)))

        for row in raw_rows:
            parsed_row = self.parse_listing_row(row)
            yield parsed_row

    def set_dataset_folder(self, _folder_path):
        assert os.path.exists(_folder_path), \
            'The dataset folder "{}" does not exist!'.format(_folder_path)

        self.__definition['dataset_folder'] = _folder_path

    def set_audio_folder(self, _folder_path):
        _dataset_folder = self.__definition['dataset_folder']
        _audio_folder = os.path.join(_dataset_folder, _folder_path)
        assert os.path.exists(_audio_folder), \
            'The audio folder "{}" does not exist!'.format(_audio_folder)

        self.__definition['audio_folder'] = _audio_folder

    def set_train_listing(self, _file_path):
        _dataset_folder = self.__definition['dataset_folder']
        _listing_path = os.path.join(_dataset_folder, _file_path)
        assert os.path.exists(_listing_path), \
            'The train listing file "{}" does not exist!'.format(_listing_path)

        self.__definition['train_listing'] = _listing_path

    def set_eval_listing(self, _file_path):
        _dataset_folder = self.__definition['dataset_folder']
        _listing_path = os.path.join(_dataset_folder, _file_path)
        assert os.path.exists(_listing_path), \
            'The eval listing file "{}" does not exist!'.format(_listing_path)

        self.__definition['eval_listing'] = _listing_path

    def generate_vocabulary(self):
        train_listing_file = os.path.join(dataset.get_definition()['dataset_folder'],
                                          dataset.get_definition()['train_listing']
                                          )
        raw_train_rows = dataset.load_listing_file(train_listing_file)
        print('Loaded {} train rows'.format(len(raw_train_rows)))

        eval_listing_file = os.path.join(dataset.get_definition()['dataset_folder'],
                                         dataset.get_definition()['eval_listing']
                                         )
        raw_eval_rows = dataset.load_listing_file(eval_listing_file)
        print('Loaded {} eval rows'.format(len(raw_eval_rows)))

        raw_rows = raw_train_rows + raw_eval_rows
        print('Loaded {} rows in total'.format(len(raw_rows)))

        parsed_sentences = [dataset.parse_listing_row(row)['sentence'] for row in raw_rows]
        _vocabulary_dict = dataset.collect_vocabulary(parsed_sentences)
        print(_vocabulary_dict)

        self.__definition['vocabulary'] = _vocabulary_dict

    def load(self):
        """
        Load the dataset definition file from disk.
        """
        # Make sure the dataset file actually exists.
        assert os.path.exists(self.__dataset_file), \
            'The dataset file "{}" does not exist!'.format(self.__dataset_file)

        # Load the definition file.
        with open(self.__dataset_file, 'r') as json_file:
            definition = json.loads(json_file.read())

        self.__definition = definition

    def save(self):
        """
        Save the dataset definition file to disk.
        """
        # Make sure the dataset file actually exists.
        assert os.path.exists(self.__dataset_file), \
            'The dataset file "{}" does not exist!'.format(self.__dataset_file)

        with open(self.__dataset_file, 'w') as json_file:
            json.dump(self.__definition, json_file, indent=2)

    def load_listing_file(self, _listing_file):
        with open(_listing_file, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ', quotechar='|')
            return list(csv_reader)

    def parse_listing_row(self, _row):
        definition = self.__definition
        audio_folder = os.path.join(
            definition['dataset_folder'],
            definition['audio_folder']
        )
        return {
            'audio_path': os.path.join(audio_folder, _row[0]),
            'sentence': _row[1],
        }

    def collect_vocabulary(self, _parsed_sentences):
        vocabulary_set = set()
        for sentence in _parsed_sentences:
            vocabulary_set.update(sentence)

        print('len(vocabulary_set)', len(vocabulary_set))
        vocabulary_dict = dict()
        vocabulary_dict['pad'] = 0
        vocabulary_dict['eos'] = 1
        for i, symbol in zip(range(2, len(vocabulary_set)), sorted(list(vocabulary_set))):
            vocabulary_dict[symbol] = i

        return vocabulary_dict

    def py_load_audio(self, _path):
        # Load the actual audio file.
        audio, sr = load_wav(_path.decode())

        return audio, sr

    def py_get_spectrograms(self, audio, sr):
        raise NotImplementedError

    def generate_normalization(self):
        # Collect decibel statistics for all the files in the train listing.
        train_listing_file = os.path.join(
            self.get_definition()['dataset_folder'],
            self.get_definition()['train_listing']
        )
        raw_train_rows = self.load_listing_file(train_listing_file)
        print('Loaded {} train rows'.format(len(raw_train_rows)))

        path_listing = [dataset.parse_listing_row(row)['audio_path'] for row in raw_train_rows]

        # (min_linear, max_linear, min_mel, max_mel)
        stats = statistics.collect_decibel_statistics(path_listing, n_threads=4)

        self.__definition['normalization'] = {
            "mel_mag_ref_db": stats[3],
            "mel_mag_max_db": stats[2],
            "linear_ref_db": stats[1],
            "linear_mag_max_db": stats[0]
        }

        return stats


if __name__ == '__main__':
    dataset = Dataset('/tmp/LJSpeech-1.1/dataset.json')
    dataset.load()

    # dataset.set_dataset_folder('/tmp/LJSpeech-1.1/')
    # dataset.set_audio_folder('wavs')
    # dataset.set_train_listing('train.csv')
    # dataset.set_eval_listing('eval.csv')
    # dataset.generate_vocabulary()
    # dataset.generate_normalization()
    # dataset.save()

    # for element in dataset.get_train_listing_generator():
    #     print(element)

    for element in dataset.get_eval_listing_generator():
        print(element)
