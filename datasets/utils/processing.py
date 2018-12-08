"""
Collection of generic helper functions for handling strings and spectrogram.
"""

import math
import os
import re
import csv
from multiprocessing.pool import ThreadPool

import numpy as np

from audio.conversion import ms_to_samples, magnitude_to_decibel, normalize_decibel, \
    inv_normalize_decibel, decibel_to_magnitude
from audio.features import linear_scale_spectrogram, mel_scale_spectrogram
from audio.io import load_wav
from audio.synthesis import spectrogram_to_wav


def utf8_to_ascii(sentence):
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


def replace_abbreviations(abbreviations, sentence):
    """
    Expand / replace abbreviations inside a string.

    Arguments:
        abbreviations (:obj:`dict` of :obj:`str`):
            Abbreviation translation dictionary.
            Every substring matching a key in the `abbreviations` dictionary is replaced with
            the key.

        sentence (str):
            String in which to expand abbreviations.
            The string is expected to only contain lowercase characters.

    Returns:
        sentence (str):
            String in which abbreviations with their expanded forms.
    """
    for abbreviation, expansion in abbreviations.items():
        # Replace abbreviation if it exists in the string.
        sentence = sentence.replace(abbreviation, expansion)

    return sentence


def filter_whitelist(sentence, whitelist_expression):
    """
    Remove all characters from a string that do not match a whitelist expression.

    Arguments:
        sentence (str):
            String to br filtered.

        whitelist_expression:
            Compiled regex pattern object which is used for whitelisting.

    Returns (str):
        Filtered string.
    """
    filtered_sentence = re.sub(whitelist_expression, '', sentence)
    return filtered_sentence


def normalize_sentence(abbreviations, sentence):
    """
    Normalize a sentence.
    Normalization includes stripping non-printable character, lower-case conversion and
    abbreviation replacements.

    Arguments:
        abbreviations (:obj:`dict` of :obj:`str`):
            Abbreviation translation dictionary.
            Every substring matching a key in the `abbreviations` dictionary is replaced with
            the key.

        sentence (str):
            String to be normalized.

    Returns (str):
        The normalized sentence.
    """
    sentence = sentence.strip()

    # Extract the transcription.
    # We do not want the sentence to contain any non ascii characters.
    sentence = utf8_to_ascii(sentence)

    # Make sentence lowercase.
    sentence = sentence.lower()

    # Replace abbreviations.
    sentence = replace_abbreviations(abbreviations, sentence)

    return sentence


def prediction_prepare_sentence(dataset_loader, whitelist_expression, sentence):
    """
    Normalize and tokenize a sentence for prediction.
    Arguments:
        dataset_loader (datasets.dataset.Dataset):
            Dataset loading helper to load the dataset with.

        whitelist_expression:
            Compiled regex pattern object which is used for whitelisting.

        sentence (str):
            String to be processed.

    Returns (np.ndarray):
        Numpy array with `dtype=np.int32` containing the normalized and tokenized sentence.
    """

    abbreviations = dict()
    sentence = normalize_sentence(abbreviations, sentence)
    sentence = filter_whitelist(sentence, whitelist_expression)

    # Tokenize the sentence.
    tokenized_sentence = dataset_loader.sentence2tokens(sentence)
    # Append the EOS token.
    tokenized_sentence.append(dataset_loader.get_eos_token())

    return np.array(tokenized_sentence, dtype=np.int32)


# def prediction_pad_sentences(tokenized_sentences):
#     # Pad sentence to the same length in order to be able to batch them in a single tensor.
#     max_length = max(sequence_lengths)
#     sentences = np.array([__py_pad_sentence(sentence, max_length) for sentence in sentences])


def split_list_proportional(_listing, train=0.8):
    """
    Split an iterable into a separate train and eval portion based on a percentual limit.

    Arguments:
        _listing (:obj:`iter`):
            Listing to be split.

        train (float):
            Float determining the portion to split an iterable into separate train and evaluation
            portions.
            `train` is expected to fulfill (`0.0 < train < 1.0`).
            Default is `0.8`.
            After the train listing is filled the remaining elements form the eval listing.
            When collecting `train` percent does not result in an integer the number of elements
            is rounded up to the next integer.

    Returns (tuple):
        Tuple of the form (train_listing, eval_listing).
    """
    assert 0.0 < train < 1.0, \
        'Training proportion must be greater 0.0 and below 1.0.'

    n_elements = len(_listing)

    assert n_elements > 0, \
        'Number of elements to split must be greater 0.'

    n_train = math.ceil(n_elements * train)
    n_eval = n_elements - n_train

    print('Splitting the {} element dataset into train: {} and eval: {}'
          .format(n_elements, n_train, n_eval))

    train_listing = _listing[:n_train]
    eval_listing = _listing[n_train:]

    assert len(train_listing) == n_train, \
        'Filling the train portion yielded less elements than required.'

    assert len(eval_listing) == n_eval, \
        'Filling the eval portion yielded less elements than required.'

    return train_listing, eval_listing


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


def load_audio(_path):
    """
    Load an audio file from disk.

    Arguments:
        _path (str):
            Path to the audio file to be loaded.

    Returns (tuple):
        audio (np.ndarray):
            Audio file.

        sampling_rate (int):
            Sampling rate.
    """
    # Load the actual audio file using plain python code.
    audio, sampling_rate = load_wav(_path)

    return audio, sampling_rate


def py_load_audio(_path):
    """
    Load an audio file from disk in an `tf.py_func` compatible way.

    Arguments:
        _path (str):
            Path to the audio file to be loaded encoded as binary string.

    Returns (tuple):
        audio (np.ndarray):
            Audio file.

        sampling_rate (int):
            Sampling rate.
    """
    print("py_load_audio", _path)
    # Load the actual audio file in an `tf.py_func` compatible way.
    audio, sampling_rate = load_audio(_path.decode())

    return audio, sampling_rate


def py_calculate_spectrogram(mel_mag_ref_db,
                             mel_mag_max_db,
                             linear_ref_db,
                             linear_mag_max_db,
                             _win_len,
                             _win_hop,
                             sampling_rate,
                             n_fft,
                             n_mels,
                             mel_fmin,
                             mel_fmax,
                             reduction,
                             audio):
    """
    Calculate normalized mel scale and linear scale magnitude spectrograms in decibel
    representation from a waveform.

    Arguments:
        mel_mag_ref_db:
        mel_mag_max_db:
        linear_ref_db:
        linear_mag_max_db:

        _win_len (float):
            Windows length in ms.

        _win_hop (float):
            Window stride in ms.

        sampling_rate (int):
            Sampleing rate.

        n_fft (int):
            FFT window length.

        n_mels (int):
            Number of Mel bands to generate.

        mel_fmin (int):
            Mel spectrum lower cutoff frequency.

        mel_fmax (int):
            Mel spectrum upper cutoff frequency.

        reduction (int):
            reduction factor `r` used for decoder target folding.

        audio (np.ndarray):
           Audio time series.
            The shape is expected to be shape=(n,).

    Returns (:obj:`tuple` of :obj:`np.ndarray`):
        mel_mag_db (np.ndarray):
            Mel scaled magnitude spectrum in decibel representation.
            The shape of the matrix will be shape(`n_mels`, t)

        linear_mag_db (np.ndarray):
            Linear scaled magnitude spectrum in decibel representation.
            The shape is of the matrix will be shape=(`1 + n_fft/2`, t)
    """
    # Window length in audio samples.
    win_len = ms_to_samples(_win_len, sampling_rate)

    # Window hop in audio samples.
    hop_len = ms_to_samples(_win_hop, sampling_rate)

    # Calculate the linear scale spectrogram.
    # Note the spectrogram shape is transposed to be (T_spec, 1 + n_fft // 2) so dense layers
    # for example are applied to each frame automatically.
    linear_spec = linear_scale_spectrogram(audio, n_fft, hop_len, win_len).T

    # Calculate the Mel. scale spectrogram.
    # Note the spectrogram shape is transposed to be (T_spec, n_mels) so dense layers for
    # example are applied to each frame automatically.
    mel_spec = mel_scale_spectrogram(audio, n_fft, sampling_rate, n_mels,
                                     mel_fmin, mel_fmax, hop_len,
                                     win_len, 1).T

    # Convert the linear spectrogram into decibel representation.
    linear_mag = np.abs(linear_spec)
    linear_mag_db = magnitude_to_decibel(linear_mag)
    linear_mag_db = normalize_decibel(linear_mag_db, linear_ref_db, linear_mag_max_db)
    # => linear_mag_db.shape = (T_spec, 1 + n_fft // 2)

    # Convert the mel spectrogram into decibel representation.
    mel_mag = np.abs(mel_spec)
    mel_mag_db = magnitude_to_decibel(mel_mag)
    mel_mag_db = normalize_decibel(mel_mag_db, mel_mag_ref_db, mel_mag_max_db)
    # => mel_mag_db.shape = (T_spec, n_mels)

    # Tacotron reduction factor.
    if reduction > 1:
        mel_mag_db, linear_mag_db = apply_reduction_padding(mel_mag_db, linear_mag_db, reduction)

    return np.array(mel_mag_db).astype(np.float32), \
           np.array(linear_mag_db).astype(np.float32)


def synthesize(linear_mag, _win_len, _win_hop, sampling_rate, n_fft, magnitude_power,
               reconstruction_iterations):
    """
    Synthesise a waveform from an linear scale magnitude spectrogram.

    Arguments:
        linear_mag (np.ndarray):
            Linear scale magnitude spectrogram to turn into a waveform.

        _win_len (float):
            Windows length in ms.

        _win_hop (float):
            Window stride in ms.

        sampling_rate (int):
            Sampling rate.

        n_fft (int):
            FFT window size.

        magnitude_power (float):
            Linear scale magnitudes are raise to the power of `magnitude_power` before
            reconstruction.

        reconstruction_iterations (int):
            The number of Griffin-Lim reconstruction iterations.

    Returns (np.ndarray):
        Audio time series.
        The shape of the returned array is shape=(n,) and the arrays dtype is np.float32.
    """
    # linear_mag = np.squeeze(linear_mag, -1)
    linear_mag = np.power(linear_mag, magnitude_power)

    win_len = ms_to_samples(_win_len, sampling_rate)
    win_hop = ms_to_samples(_win_hop, sampling_rate)

    print('Spectrogram inversion ...')
    return spectrogram_to_wav(linear_mag,
                              win_len,
                              win_hop,
                              n_fft,
                              reconstruction_iterations)


def post_process_spectrograms(_spectrograms, dataset_loader, synthesis_fn, n_synthesis_threads):
    """
    Post-process spectrograms and synthesise waveforms.
    Post-processing includes reverting the normalization process and transforming spectrograms
    from decibel representation into magnitude spectrograms.

    Arguments:
        _spectrograms (:obj:`list` of :obj:`np.ndarray`):
            List of spectrograms to post-process.

        dataset_loader (datasets.dataset.Dataset):
            Dataset loading helper to load the dataset with.

        synthesis_fn (callable):
            Function to call for each spectrogram to be synthezised.
            `synthesis_fn` is passed a linear magnitude spectrogram as an argument `linear_mag`.
            `synthesis_fn` is expected to return a synthezised waveform in form of an
            `np.ndarrary` object.

        n_synthesis_threads (int):
            Number of thread to use for synthesis.

    Returns (:obj:`list` of :obj:`np.ndarray`):
        List of synthesised waveforms.
    """
    normalization_params = dataset_loader.get_normalization()
    # Apply Griffin-Lim to all spectrogram's to get the waveforms.
    normalized = list()
    for spectrogram in _spectrograms:
        print('Reverse spectrogram normalization ...', spectrogram.shape)
        linear_mag_db = inv_normalize_decibel(spectrogram.T,
                                              normalization_params['mel_mag_ref_db'],
                                              normalization_params['mel_mag_max_db'])

        linear_mag = decibel_to_magnitude(linear_mag_db)
        normalized.append(linear_mag)

    specs = normalized

    # Synthesize waveforms from the spectrograms.
    pool = ThreadPool(n_synthesis_threads)
    wavs = pool.map(synthesis_fn, specs)
    pool.close()
    pool.join()

    return wavs


def derive_bucket_boundaries(dataset_generator, key, n_buckets):
    """
    Derive optimal bucket boundaries from data.

    Arguments:
        dataset_generator (:obj:`iter` of :obj:`dict`):
            Generator to collect data from for deriving the buckets.

        key (str):
            Key of the column which should be used to derive the boundaries.

        n_buckets (int):
            Number of buckets to sort data into.
            If it is not possible to create as meany buckets as requested, less buckets are created.

    Returns (:obj:`list` of :obj:`int`):
        Bucket boundaries for use in tensorflows `grouping.bucket_by_sequence_length`.
    """
    element_lengths = [row[key] for row in dataset_generator]

    # Get the total number of samples in the dataset.
    n_samples = len(element_lengths)

    # Sort sequence lengths in order to slice them into buckets that contain sequences of roughly
    # equal length.
    sorted_sentence_lengths = np.sort(element_lengths)

    if n_samples < n_buckets:
        raise AssertionError('The number of entries loaded is smaller than the number of '
                             'buckets to be created. Automatic calculation of the bucket '
                             'boundaries is not possible.')

    # Slice the sorted lengths into equidistant sections and use the first element of a slice as
    # the bucket boundary.
    bucket_step = n_samples // n_buckets
    bucket_boundaries = sorted_sentence_lengths[::bucket_step]

    # Throw away the first and last bucket boundaries since the bucketing algorithm automatically
    # adds two surrounding ones.
    bucket_boundaries = bucket_boundaries[1:-1].tolist()

    # Remove duplicate boundaries from the list.
    bucket_boundaries = sorted(list(set(bucket_boundaries)))

    print('bucket_boundaries', bucket_boundaries)
    print('n_buckets: {} + 2'.format(len(bucket_boundaries)))

    return bucket_boundaries


def py_load_processed_features(audio_path):
    """
    Load pre-processed features from disk in an `tf.py_func` compatible way.

    Arguments:
        audio_path (str):
            Path to the audio file to load the corresponding features for.
            `audio_path` is expected to be a binary string.

    Returns (tuple):
        mel_mag_db (np.ndarray):
            Mel scale spectrogram.

        linear_mag_db (np.ndarray):
            Linear scale spectrogram.
    """
    file_path = os.path.splitext(audio_path.decode())[0]

    # Load features from disk.
    data = np.load('{}.npz'.format(file_path))

    return data['mel_mag_db'], data['linear_mag_db']


def write_listing_csv(_listing, _listing_file):
    """
    Write a .csv listing file to disk.

    Arguments:
        _listing (iter):
            Iterator over value tuples forming a single row to write.

        _listing_file (str):
            path to the listing file to write.
    """
    # Open a .csv file for writing.
    with open(_listing_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='|', quotechar='|')
        for row in _listing:
            csv_writer.writerow(row)
