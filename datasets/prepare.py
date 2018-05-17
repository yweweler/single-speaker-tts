import numpy as np
import os
from datasets.pavoque import PAVOQUEDatasetHelper
from datasets.lj_speech import LJSpeechDatasetHelper


def prepare_dataset(dataset):
    # Load alĺ sentences and the corresponding audio file paths.
    sentences, sentence_lengths, wav_paths = dataset.load(min_len=None, max_len=None)

    # Get the total number of samples in the dataset.
    n_samples = len(sentence_lengths)

    print('Loaded {} dataset entries.'.format(n_samples))

    for wav_path in wav_paths:
        # Load and process the audio file.
        mel_mag_db, linear_mag_db = dataset.load_audio(wav_path.encode())

        # Extract the file path and file name without the extension.
        file_path = os.path.splitext(wav_path)[0]

        # Create the target file path.
        out_path = '{}.npz'.format(file_path)

        # Save the audio file as a numpy .npz file.
        np.savez(out_path, mel_mag_db=mel_mag_db, linear_mag_db=linear_mag_db)


if __name__ == '__main__':
    init_char_dict = {
        'pad': 0,  # padding
        'eos': 1,  # end of sequence
    }
    dataset = PAVOQUEDatasetHelper(dataset_folder='/home/yves-noel/downloads/PAVOQUE',
                                   char_dict=init_char_dict,
                                   fill_dict=True)

    prepare_dataset(dataset)
