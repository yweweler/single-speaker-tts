"""
Generation of dataset listing files for the LJ Speech dataset in version 1.1.
The code creates the train.csv and eval.csv listings from the original metadata.csv file.
"""

import csv
import random
import re

from datasets.utils.processing import normalize_sentence, \
    split_list_proportional, filter_whitelist, write_listing_csv


def parse_line(_line):
    """
    Parse values from a listing row of the metadata.csv file.

    Arguments:
        _line (:obj:`tuple` of :obj:`str`):
            Tuple containing the split values of a single line.

    Returns (dict):
        Parsed values in the form of a dictionary.
        The contained keys are ['id', 'transcription', 'normalized_transcription'].
    """
    return {
        'id': _line[0],
        'transcription': _line[1],
        'normalized_transcription': _line[2]
    }


def pre_process_row(_abbreviations, _whitelist_expression, _row):
    """
    Pre-process a parsed row from metadata.csv.

    Arguments:
        _abbreviations (:obj:`dict` of :obj:`str`):
            Abbreviation translation dictionary.
            Every substring matching a key in the `abbreviations` dictionary is replaced with
            the key.

        _whitelist_expression:
            Compiled regex pattern object which is used for whitelisting.

        _row (dict):
            Parsed row.

    Returns (dict):
            Post-processed and augmented values in the form of a dictionary.
            The contained keys are ['id', 'audio_file', 'sentence'].
    """
    _id = _row['id']
    _sentence = _row['normalized_transcription']

    sentence = normalize_sentence(_abbreviations, _sentence)
    sentence = filter_whitelist(sentence, _whitelist_expression)

    audio_file = '{}.wav'.format(_id)

    return {
        'id': _id,
        'audio_file': audio_file,
        'sentence': sentence
    }


def pre_process_listing(_abbreviations, _whitelist_expression, _listing_file):
    """
    Load the metadata.csv file and parse its rows.

    Arguments:
        _abbreviations (:obj:`dict` of :obj:`str`):
            Abbreviation translation dictionary.
            Every substring matching a key in the `abbreviations` dictionary is replaced with
            the key.

        _whitelist_expression:
            Compiled regex pattern object which is used for whitelisting.

        _listing_file (str):
            Path to the metadata.csv listing file.

    Returns (list):
        List of parsed and pre-processed listing rows.
    """
    _processed_rows = list()

    # Read all lines from the csv file and split them based on a delimiter.
    with open(_listing_file, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='|', quotechar='|')
        csv_lines = list(csv_reader)

    for line in csv_lines:
        # Parse a line.
        parsed_row = parse_line(line)
        # Pre-process values and augment them.
        processed_row = pre_process_row(_abbreviations, _whitelist_expression, parsed_row)
        _processed_rows.append(processed_row)

    return _processed_rows


def write_listing(_listing, _listing_file):
    """
    Write a listing file to disk.

    Arguments:
        _listing (:obj:`list` of :obj:`dict`):
            List of parsed rows to be written to a listing file.

        _listing_file (str):
            Path to the listing file to be written.
    """
    # Select the values to be written.
    values_to_write = (
        (row['audio_file'], row['sentence']) for row in _listing
    )

    # Write the file to disk.
    write_listing_csv(values_to_write, _listing_file)


def generate():
    """
    Generation of dataset listing files for the LJ Speech dataset.
    """
    # Path to the original LJ Speech listing file.
    source_listing_file = '/tmp/LJSpeech-1.1/metadata.csv'

    # Path to the training listing file.
    train_listing_file = '/tmp/LJSpeech-1.1/train.csv'

    # Path to the evaluation listing file.
    eval_listing_file = '/tmp/LJSpeech-1.1/eval.csv'

    # Abbreviation expansion lookup table.
    abbreviations = {
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
        'ft.': 'fort'
    }

    # Regular expression to select characters for whitelist filtering.
    whitelist_expression = re.compile(r'[^a-z !?,\-:;()"\']+')

    # Percentual portion of metadata.csv to use for the training listing.
    train_proportion = 0.916  # 12000 train, 1100 eval

    # Load, parse and pre-process metadata.csv.
    processed_rows = pre_process_listing(abbreviations, whitelist_expression, source_listing_file)

    # Shuffle the loaded rows.
    random.shuffle(processed_rows)

    # Split metadata.csv into separate train and eval portions.
    train_listing, eval_listing = split_list_proportional(processed_rows, train_proportion)

    # Write the generated train listing to disk.
    write_listing(train_listing, train_listing_file)

    # Write the generated eval listing to disk.
    write_listing(eval_listing, eval_listing_file)


if __name__ == '__main__':
    generate()
