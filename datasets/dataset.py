"""
Defines and implements a basic dataset.
"""

import csv
import json
import os

import numpy as np

from datasets.utils import statistics


class Dataset:
    """
    Basic dataset that can be loaded and saved.

    The dataset consists of a dataset definition file `.json` and separate `.csv` files
    containing listings for the training and evaluation portions.
    """

    def __init__(self, dataset_file):
        # Path ot the dataset definition file.
        self.__dataset_file = dataset_file

        # Dataset definition.
        self.__definition = dict()

        # Inverse version of the vocabulary dictionary from the dataste definition.
        # This translates integer tokens back to symbols.
        self.__reverse_vocabulary = None

        # Loaded and processed training and evaluation listings.
        self.__train_listing = None
        self.__eval_listing = None

    def get_definition(self):
        """
        Get the dataset definition.

        Returns (dict):
            Dataset definition dictionary
        """
        return self.__definition

    def get_vocabulary(self):
        """
        Get the dataset vocabulary.

        Returns (dict):
            Vocabulary dictionary of the dataset.
            It translates symbols into integer tokens.
        """
        return self.get_definition()['vocabulary']

    def get_normalization(self):
        """
        Get the feature normalization parameters.

        Returns (dict):
            Dictionary containing the feature normalization parameters.
        """
        return self.get_definition()['normalization']

    def get_reverse_vocabulary(self):
        """
        Get the reverse vocabulary.
        This translates integer tokens back to symbols.

        Returns (dict):
            Reverse vocabulary dictionary.
        """
        return self.__reverse_vocabulary

    def get_eos_token(self):
        """
        Get the End-Of-Sequence (EOS) token from the vocabulary.

        Returns (int):
            EOS token.
        """
        return self.get_vocabulary()['eos']

    def get_pad_token(self):
        """
        Get the Padding (PAD) token from the vocabulary.

        Returns (int):
            PAD token.
        """
        return self.get_vocabulary()['pad']

    def sentence2tokens(self, sentence):
        """
        Convert each character of a string into its corresponding dictionary id.

        Arguments:
            sentence (str):
                Sentence to be converted.

        Returns:
            ids (:obj:`list` of int):
                List containing the id's corresponding to each character in the input sentence.
                With a length of: `len(ids) == len(sentence)`.

        Raises:
            KeyError:
                If a character was encountered that is not contained in the translation dictionary.
        """
        vocabulary = self.get_vocabulary()
        return [vocabulary[char] for char in sentence]

    def tokens2sentence(self, idx):
        """
        Convert a list of dictionary id's into the corresponding characters.

        Arguments:
            idx (:obj:`list` of int):
                List of dictionary id's to be converted.

        Returns:
            sentence (str):
                Converted sentence.
                With a length of: `len(sentence) == len(idx)`.

        Raises:
            KeyError:
                If a id was encountered that is not contained in the translation dictionary.
        """
        reverse_vocabulary = self.get_reverse_vocabulary()
        return ''.join([reverse_vocabulary[_id] for _id in idx])

    def get_train_listing_generator(self, max_samples=None):
        """
        Create a generator over the parsed training listing rows.

        Arguments:
            max_samples (int):
                Number of samples after which to stop generating.
                Default is `None`, meaning that all available rows should be returned.

        Yields (dict):
            Parsed listing row as a dictionary.
            The keys are: ['audio_path', sentence, 'tokenized_sentence',
            'tokenized_sentence_length'].
        """
        for i, _element in enumerate(self.__train_listing):
            if max_samples is not None:
                if i + 1 > max_samples:
                    return

            yield _element

    def get_eval_listing_generator(self, max_samples=None):
        """
        Create a generator over the parsed evaluation listing rows.

        Arguments:
            max_samples (int):
                Number of samples after which to stop generating.
                Default is `None`, meaning that all available rows should be returned.

        Yields (dict):
            Parsed listing row as a dictionary.
            The keys are: ['audio_path', sentence, 'tokenized_sentence',
            'tokenized_sentence_length'].
        """
        for i, _element in enumerate(self.__eval_listing):
            if max_samples is not None:
                if i + 1 > max_samples:
                    return

            yield _element

    def set_dataset_folder(self, _folder_path):
        """
        Set the dataset base folder in the dataset definition.

        Arguments:
            _folder_path (str):
                Path to the dataset base folder.
        """
        assert os.path.exists(_folder_path), \
            'The dataset folder "{}" does not exist!'.format(_folder_path)

        self.__definition['dataset_folder'] = _folder_path

    def set_audio_folder(self, _folder_path):
        """
        Set the audio folder in the dataset definition.

        Arguments:
            _folder_path (str):
                Path to the audio folder relative from the dataset base folder.
        """
        _dataset_folder = self.__definition['dataset_folder']
        _audio_folder = os.path.join(_dataset_folder, _folder_path)
        assert os.path.exists(_audio_folder), \
            'The audio folder "{}" does not exist!'.format(_audio_folder)

        self.__definition['audio_folder'] = _audio_folder

    def set_train_listing_file(self, _file_path):
        """
        Set the path to the train listing file in the dataset definition.

        Arguments:
            _file_path (str):
                Path to the listing file relative from the dataset baser folder.
        """
        _dataset_folder = self.__definition['dataset_folder']
        _listing_path = os.path.join(_dataset_folder, _file_path)
        assert os.path.exists(_listing_path), \
            'The train listing file "{}" does not exist!'.format(_listing_path)

        self.__definition['train_listing'] = _listing_path

    def set_eval_listing_file(self, _file_path):
        """
        Set the path to the eval listing file in the dataset definition.

        Arguments:
            _file_path (str):
                Path to the listing file relative from the dataset baser folder.
        """
        _dataset_folder = self.__definition['dataset_folder']
        _listing_path = os.path.join(_dataset_folder, _file_path)
        assert os.path.exists(_listing_path), \
            'The eval listing file "{}" does not exist!'.format(_listing_path)

        self.__definition['eval_listing'] = _listing_path

    def generate_vocabulary(self):
        all_rows = self.__train_listing + self.__eval_listing

        parsed_sentences = [row['sentence'] for row in all_rows]
        _vocabulary_dict = dataset.collect_vocabulary(parsed_sentences)
        print(_vocabulary_dict)

        self.__definition['vocabulary'] = _vocabulary_dict
        self.__generate_reverse_vocabulary()

    def __generate_reverse_vocabulary(self):
        # Initialize reverse_dictionary in case a user defined dictionary was passed.
        self.__reverse_vocabulary = dict()
        for _token, _id in self.get_vocabulary().items():
            self.__reverse_vocabulary[_id] = _token

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
        self.__generate_reverse_vocabulary()

    def save(self):
        """
        Save the dataset definition file to disk.
        """
        # Make sure the dataset file actually exists.
        assert os.path.exists(self.__dataset_file), \
            'The dataset file "{}" does not exist!'.format(self.__dataset_file)

        with open(self.__dataset_file, 'w') as json_file:
            json.dump(self.__definition, json_file, indent=2)

    def load_listings(self):
        train_listing_file = os.path.join(
            self.get_definition()['dataset_folder'],
            self.get_definition()['train_listing']
        )
        parsed_train_rows = self.__load_listing_file(train_listing_file)
        self.__train_listing = parsed_train_rows
        print('Loaded {} train rows'.format(len(parsed_train_rows)))

        eval_listing_file = os.path.join(
            self.get_definition()['dataset_folder'],
            self.get_definition()['eval_listing']
        )
        parsed_eval_rows = self.__load_listing_file(eval_listing_file)
        self.__eval_listing = parsed_eval_rows
        print('Loaded {} eval rows'.format(len(parsed_eval_rows)))

    def __load_listing_file(self, _listing_file):
        with open(_listing_file, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='|', quotechar='|')
            csv_rows = list(csv_reader)

            parsed_rows = list()
            for row in csv_rows:
                parsed_row = self.__parse_listing_row(row)

                # Tokenize sentence.
                sentence = parsed_row['sentence']
                tokenized_sentence = self.sentence2tokens(sentence)
                tokenized_sentence.append(self.get_eos_token())

                # Get tokenized sentence length.
                tokenized_sentence_length = len(tokenized_sentence)

                parsed_row.update({
                    'tokenized_sentence': np.array(tokenized_sentence, dtype=np.int32),
                    'tokenized_sentence_length': tokenized_sentence_length
                })
                parsed_rows.append(parsed_row)

        return parsed_rows

    def __parse_listing_row(self, _row):
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

    def generate_normalization(self, n_threads=1):
        path_listing = [row['audio_path'] for row in self.__train_listing]

        # (min_linear, max_linear, min_mel, max_mel)
        stats = statistics.collect_decibel_statistics(path_listing, n_threads=n_threads)

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
    dataset.load_listings()

    # dataset.set_dataset_folder('/tmp/LJSpeech-1.1/')
    # dataset.set_audio_folder('wavs')
    # dataset.set_train_listing_file('train.csv')
    # dataset.set_eval_listing_file('eval.csv')
    # TODO: Fix cyclic dependency between `load_listings` and `generate_vocabulary`.
    # dataset.load_listings()
    # dataset.generate_vocabulary()
    # dataset.generate_normalization()
    # TODO: Implement automatic feature pre-calculation.
    # dataset.save()

    # for element in dataset.get_train_listing_generator():
    #     print(element)

    for element in dataset.get_eval_listing_generator():
        print(element)
