import abc
import os

import numpy as np


class DatasetHelper:
    """
    Dataset loading helper base class.
    """

    def __init__(self, dataset_folder, char_dict, fill_dict):
        """
        DatasetHelper constructor.

        Arguments:
            dataset_folder (str):
                Path to the dataset folder.

            char_dict (dict):
                char2idx translation dictionary.
                It is required that the passed dictionary always contains two predefined tokens.
                These tokens are the padding token <PAD> with key="pad" and the end of sequence
                token <EOS> with key="eos".

                In case `fill_dict` is True the dictionary is automatically populated with
                symbols encountered while loading the dataset. Other wise the passed dictionary
                is expected to contain all symbols that are known to be in the dataset.

                The reverse (idx2char) dictionary is generated automatically.

            fill_dict (boolean):
                Flag controlling whether the cha2idx and idx2char translation dictionaries should
                be updated using the loaded data.
                If True the dictionaries are augmented with characters from the dataset.
                If False the dictionary is expected to be complete and will not be updated.
        """
        self._dataset_folder = dataset_folder
        self._char2idx_dict = char_dict
        self._fill_dict = fill_dict
        self._abbreviations = dict()

        # Dataset statistics.
        self._statistics = dict()

        # Initialize idx_dict in case a user defined dictionary was passed.
        self._idx2char_dict = dict()
        for char, _id in self._char2idx_dict.items():
            self._idx2char_dict[_id] = char

    def sent2idx(self, sentence):
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
        return [self._char2idx_dict[char] for char in sentence]

    def idx2sent(self, idx):
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
        return ''.join([self._idx2char_dict[_id] for _id in idx])

    def utf8_to_ascii(self, sentence):
        """
        Convert a UTF-8 encoded string into ASCII representation.

        UTF-8 symbols that can not be converted into ASCII representation are dropped.

        Arguments:
            sentence (str):
                UTF-8 encoded string to be converted.

        Returns:
            ascii_sentence (str):
                ASCII encoded string.
        """
        # Convert the UTF-8 sentence into bytes representation for decoding.
        utf_sentence = bytes(sentence, 'utf-8')

        # Decode the byte representation into ASCII representation.
        ascii_sentence = utf_sentence.decode('ascii', errors='ignore')

        return ascii_sentence

    def update_char_dict(self, sentence):
        """
        Populate the char2idx and idx2char translation dictionaries with characters from a sentence.

        Arguments:
            sentence (str):
                String whose characters are used to update the translation dictionaries.
        """
        # Update character dictionary with the chars contained in the sentence.
        for char in sentence:
            if char not in self._char2idx_dict.keys():
                # Character is not contained in the dictionary, we need to add it.
                _id = len(self._char2idx_dict.keys())
                self._char2idx_dict[char] = _id
                self._idx2char_dict[_id] = char

    def replace_abbreviations(self, sentence):
        """
        Expand / replace abbreviations inside a string.

        Arguments:
            sentence (str):
                String in which to expand abbreviations.
                The string is expected to only contain lowercase characters.

        Returns:
            sentence (str):
                String in which abbreviations with their expanded forms.
        """
        for abbreviation, expansion in self._abbreviations.items():
            # Replace abbreviation if it exists in the string.
            sentence = sentence.replace(abbreviation, expansion)

        return sentence

    def process_sentences(self, sentences):
        """
        Processes a list of sentences.

        The processing steps applied to each sentence are:
            - Convert sentence to lowercase.
            - Expand / replace abbreviations.
            - If requested (`_fill_dict == True`) update the char2idx and idx2char dictionaries.
            - Convert sentence into a sequence of dictionary id's.
            - Append the EOS token.

        Arguments:
            sentences (:obj:`list` of str):
                List of sentences to be processed.

        Returns:
            id_sequences (:obj:`list` of bytes):
                List of Python bytes exhibiting a copy of a numpy array containing the integer id's.
                Each entry resembles `np.array(sentence_xy_ids, dtype=np.int32).tostring()`.
                Note that is is done for better feeding into tensorflow tensors.
                Its length fulfills `len(sequence_lengths) == len(sentences)`.
            sequence_lengths (:obj:`list` of int):
                List containing the produced sequence length for each sentence. This sequence
                lengths include the EOS tokens. Its length fulfills
                `len(sequence_lengths) == len(sentences)`.
        """
        id_sequences = list()
        sequence_lengths = list()

        # Reset word statistics.
        self._statistics['n_words_total'] = 0
        self._statistics['n_words_unique'] = 0
        self._statistics['n_words_clip_avg'] = 0

        # Reset character statistics.
        self._statistics['n_chars_total'] = 0
        self._statistics['n_chars_unique'] = 0
        self._statistics['n_chars_clip_avg'] = 0

        # Unique word lookup set.
        word_set = set()

        # Unique character lookup set.
        character_set = set()

        # TODO: Refactor.
        def __update_word_statistics(sentence):
            words = sentence.split(' ')
            self._statistics['n_words_total'] += len(words)
            word_set.update(words)

        # TODO: Refactor.
        def __update_character_statistics(sentence):
            self._statistics['n_chars_total'] += len(sentence)
            character_set.update(list(sentence))

        for sentence in sentences:
            # Make sentence lowercase.
            sentence = sentence.lower()

            # Collect word statistics.
            __update_word_statistics(sentence)

            # Collect character statistics.
            __update_character_statistics(sentence)

        self._statistics['n_words_unique'] = len(word_set)
        self._statistics['n_chars_unique'] = len(character_set)
        self._statistics['n_words_clip_avg'] = self._statistics['n_words_total'] / len(sentences)
        self._statistics['n_chars_clip_avg'] = self._statistics['n_chars_total'] / len(sentences)

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
            sequence_lengths.append(len(idx))

        return id_sequences, sequence_lengths

    @abc.abstractmethod
    def load(self, max_samples, min_len, max_len, listing_file_name):
        """
        Load samples from the dataset.

        Arguments:
            max_samples (int):
                The maximal number of samples to load.

            min_len (int):
                Minimal length a sentence has to have to be loaded. If `len(sentence) < min_len`
                loading the sentence is skipped.
                If None the minimal sentence length is not checked.

            max_len (int):
                Maximal length a sentence is allowed to to be loaded. If `len(sentence) > max_len`
                loading the sentence is skipped.
                If None the maximal sentence length is not checked.

            listing_file_name (string):
                Filename of the file containing the metadata to be loaded.
                This might be used to load different parts of the dataset
                (train, test or data with different speaking styles for example).

        Returns:
            (id_sentences, sentence_lengths, file_paths):
                id_sequences (:obj:`list` of bytes):
                    List of Python bytes exhibiting a copy of a numpy array containing the integer id's.
                    Each entry resembles `np.array(sentence_xy_ids, dtype=np.int32).tostring()`.
                    Note that is is done for better feeding into tensorflow tensors.
                    Its length fulfills `len(sequence_lengths) == len(sentences)`.
                sequence_lengths (:obj:`list` of int):
                    List containing the produced sequence length for each sentence. This sequence
                    lengths include the EOS tokens. Its length fulfills
                    `len(sequence_lengths) == len(sentences)`.
                file_paths (:obj:`list` of str):
                    List of paths to the audio recordings of each sentence.
                    Its length fulfills `len(sequence_lengths) == len(sentences)`.
        """
        raise NotImplementedError

    @staticmethod
    def cache_precalculated_features(wav_paths):
        """
        Loads pre-calculated feature from disk and holds them in RAM.

        Arguments:
            wav_paths (:obj:`list` of str):
                Paths to the waveform files for which to load pre-processed features.

        Returns:
            cache (dict):
                Dictionary containing the cached features with the file paths as keys for access.
        """
        __cache = dict()

        for wav_path in wav_paths:
            file_path = os.path.splitext(wav_path)[0]
            __cache[file_path] = np.load('{}.npz'.format(file_path))

        return __cache

    @staticmethod
    @abc.abstractstaticmethod
    def load_audio(file_path):
        """
        Load a audio recording from file and calculate features.

        Arguments:
            file_path (str):
                Path to the audio file to be loaded.

        Returns:
            (mel_mag_db, linear_mag_db):
                mel_mag_db (np.ndarray):
                    Mel. scale magnitude spectrogram. The arrays dtype is np.float32 and the
                    shape is shape=(T_spec, n_mels).
                linear_mag_db (np.ndarray):
                    linear. scale magnitude spectrogram. The arrays dtype is np.float32 and the
                    shape is shape=(T_spec, 1 + n_fft // 2).
        """
        raise NotImplementedError

    def pre_compute_features(self, paths):
        """
        Loads all audio files from the dataset, computes features and saves these pre-computed
        features as numpy .npz files to disk.

        For example: The features of an audio file <path>/<filename>.wav are saved next to the
        audio file in <path>/<filename>.npz.

        Arguments:
            paths (:obj:`list` of :obj:`str`):
                File paths for all audio files of the dataset to pre-compute features for.
        """
        # Get the total number of samples in the dataset.
        n_samples = len(paths)

        print('Loaded {} dataset entries.'.format(n_samples))

        for wav_path in paths:
            # Load and process the audio file.
            mel_mag_db, linear_mag_db = self.load_audio(wav_path.encode())

            # Extract the file path and file name without the extension.
            file_path = os.path.splitext(wav_path)[0]

            # Create the target file path.
            out_path = '{}.npz'.format(file_path)
            print('Writing: "{}"'.format(out_path))

            # Save the audio file as a numpy .npz file.
            np.savez(out_path, mel_mag_db=mel_mag_db, linear_mag_db=linear_mag_db)

    @staticmethod
    def apply_reduction_padding(mel_mag_db, linear_mag_db, reduction_factor):
        # TODO: Refactor function, since it does also reshape the data to the reduced shape.
        """
        Adds zero padding frames in the time axis to the Mel. scale and linear scale spectrogram's
        such that the number of frames is a multiple of the `reduction_factor`.

        Arguments:
            mel_mag_db (np.ndarray):
                Mel scale spectrogram to be reduced. The shape is expected to be
                shape=(T_spec, n_mels), with T_spec being the number of frames.

            linear_mag_db (np.ndarray):
                Linear scale spectrogram to be reduced. The shape is expected to be
                shape=(T_spec, 1 + n_fft // 2), with T_spec being the number of frames.

            reduction_factor (int):
                The number of consecutive frames.

        Returns:
            (mel_mag_db, linear_mag_db):
                mel_mag_db (np.ndarray):
                    Padded Mel. scale spectrogram.
                linear_mag_db (np.ndarray):
                    padded linear scale spectrogram.
        """
        # Number of frames in the spectrogram's.
        n_frames = mel_mag_db.shape[0]

        # Make sure the number of frames is a multiple of `reduction_factor` for reduction.
        if (n_frames % reduction_factor) != 0:
            # Calculate the number of padding frames that have to be added.
            n_padding_frames = reduction_factor - (n_frames % reduction_factor)

            # Add padding frames containing zeros to the mel spectrogram.
            mel_mag_db = np.pad(mel_mag_db, [[0, n_padding_frames], [0, 0]], mode="constant")

            # Pad the linear spectrogram since it has to have the same num. of frames.
            linear_mag_db = np.pad(linear_mag_db, [[0, n_padding_frames], [0, 0]], mode="constant")

        # Reduce `reduction_factor` consecutive frames into a single frame.
        mel_mag_db = mel_mag_db.reshape((-1, mel_mag_db.shape[1] * reduction_factor))
        linear_mag_db = linear_mag_db.reshape((-1, linear_mag_db.shape[1] * reduction_factor))

        return mel_mag_db, linear_mag_db
