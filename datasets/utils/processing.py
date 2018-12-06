import math
import re

import numpy as np

from audio.conversion import ms_to_samples, magnitude_to_decibel, normalize_decibel
from audio.features import linear_scale_spectrogram, mel_scale_spectrogram
from audio.io import load_wav


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
    filtered_sentence = re.sub(whitelist_expression, '', sentence)
    return filtered_sentence


def normalize_sentence(abbreviations, sentence):
    sentence = sentence.strip()

    # Extract the transcription.
    # We do not want the sentence to contain any non ascii characters.
    sentence = utf8_to_ascii(sentence)

    # Make sentence lowercase.
    sentence = sentence.lower()

    # Replace abbreviations.
    sentence = replace_abbreviations(abbreviations, sentence)

    return sentence


def split_list_proportional(_listing, train=0.8):
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


def py_load_audio(_path):
    # Load the actual audio file.
    audio, sr = load_wav(_path.decode())

    return audio, sr


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
