import numpy as np
import math
import re


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
