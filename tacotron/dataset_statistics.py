from datasets.statistics import collect_decibel_statistics
from tacotron.params.dataset import dataset_params
import os

if __name__ == '__main__':
    dataset = dataset_params.dataset_loader(dataset_folder=dataset_params.dataset_folder,
                                            char_dict={
                                                'pad': 0,  # padding
                                                'eos': 1,  # end of sequence
                                            },
                                            fill_dict=True)

    if not os.path.exists(dataset_params.dataset_folder):
        print("Dataset folder '{}' could not be found.".format(dataset_params.dataset_folder))
        exit(1)

    print("Dataset: {}".format(dataset_params.dataset_folder))
    print("Loading dataset ...")
    ids, lens, paths = dataset.load()

    # Collect char vocabulary after the loader processed and normalized the trancsscripts.
    print("Dataset vocabulary:")
    vocabulary = dataset._char2idx_dict
    sorted_by_value = sorted(vocabulary.items(), key=lambda kv: kv[1])
    print("vocabulary_dict={")
    for k, v in sorted_by_value:
        print("    '{}': {}".format(k, v))
    print("},")
    print("vocabulary_size={}".format(len(sorted_by_value)))
    print("\n\n")

    # Collect decibel statistics for all the files.
    print("Collecting decibel statistics for {} files ...".format(len(paths)))
    min_linear_db, max_linear_db, min_mel_db, max_mel_db = collect_decibel_statistics(paths)

    print("mel_mag_ref_db = ", max_mel_db)
    print("mel_mag_max_db = ", min_mel_db)
    print("linear_ref_db = ", max_linear_db)
    print("linear_mag_max_db = ", min_linear_db)
