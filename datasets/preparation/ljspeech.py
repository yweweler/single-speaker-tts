import csv
import re

from datasets.utils.processing import normalize_sentence, split_list_proportional, filter_whitelist


def parse_line(_line):
    return {
        'id': _line[0],
        'transcription': _line[1],
        'normalized_transcription': _line[2]
    }


def pre_process_row(_abbreviations, whitelist_expression, _row):
    _id = _row['id']
    _sentence = _row['normalized_transcription']

    sentence = normalize_sentence(_abbreviations, _sentence)
    sentence = filter_whitelist(sentence, whitelist_expression)
    return {
        'id': _id,
        'sentence': sentence
    }


# TODO: Write some generic loader code that also defines the delimiters to use.
def pre_process_listing(abbreviations, whitelist_expression, _listing_file):
    with open(_listing_file, 'r', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='|', quotechar='|')
        csv_lines = list(csv_reader)

        processed_rows = list()
        for line in csv_lines:
            parsed_row = parse_line(line)
            processed_row = pre_process_row(abbreviations, whitelist_expression, parsed_row)
            processed_rows.append(processed_row)

    return processed_rows


# TODO: Write some generic writer code that also defines the delimiters to use.
def write_listing(_listing, _listing_file):
    with open(_listing_file, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='|', quotechar='|')
        for processed_row in _listing:
            csv_writer.writerow(
                (processed_row['id'], processed_row['sentence'])
            )


if __name__ == '__main__':
    source_listing_file = '/tmp/LJSpeech-1.1/metadata.csv'
    train_listing_file = '/tmp/LJSpeech-1.1/train.csv'
    eval_listing_file = '/tmp/LJSpeech-1.1/eval.csv'
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
    whitelist_expression = re.compile(r'[^a-z !?,\-:;()"\']+')
    train_proportion = 0.916  # 12000 train, 1100 eval

    processed_rows = pre_process_listing(abbreviations, whitelist_expression, source_listing_file)
    train_listing, eval_listing = split_list_proportional(processed_rows, train_proportion)

    write_listing(train_listing, train_listing_file)
    write_listing(eval_listing, eval_listing_file)
