"""
This script generates <wav_path> | <sentence> listings for each of the speaking styles of the
PAVOQUE dataset.
"""

import os

# A dictionary storing the sentences and record ids for each recorded speaking style.
recordings = {
    'poker': {
        # <id>: (<sentence>, <wav_file_path>)
    },
    'happy': {
        # <id>: (<sentence>, <wav_file_path>)
    },
    'neutral': {
        # <id>: (<sentence>, <wav_file_path>)
    },
    'angry': {
        # <id>: (<sentence>, <wav_file_path>)
    },
    'sad': {
        # <id>: (<sentence>, <wav_file_path>)
    },
}

base_path = 'processed'
wav_folder = os.path.join(base_path, 'wav')
text_folder = os.path.join(base_path, 'text')

wav_listing_file = os.path.join(base_path, 'wav.txt')

with open(wav_listing_file, 'r') as wav_files:
    for wav_path in wav_files:
        wav_path = wav_path.replace('\n', '')
        line = wav_path.replace('.wav', '')

        # Extract record id and the recording style.
        _id, _mode = line.split('-', maxsplit=1)

        # Build the path to the transcription file.
        text_file_path = os.path.join(text_folder, '{}.txt'.format(_id))
        assert os.path.isfile(text_file_path), 'Could not find requested file "{}".'\
            .format(text_file_path)

        # Load transcription.
        with open(text_file_path, 'r') as text_file:
            text = text_file.readline()
            text = text.replace('\n', '')

        wav_file_path = os.path.join('wav', wav_path)

        recordings[_mode][_id] = (text, wav_file_path)

# Write "<id> | <sentence>" pairs into separate files for each speaking style.
for style in recordings.keys():
    file_name = '{}.txt'.format(style)
    with open(os.path.join(base_path, file_name), 'w') as listing:
        for _id, (_sentence, _wav_path) in recordings[style].items():
            print(_id, _sentence, _wav_path)

            # The id and the sentence are delimited by " | ".
            line = '{} | {}\n'.format(_wav_path, _sentence)
            listing.write(line)
