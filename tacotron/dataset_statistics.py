from datasets.statistics import collect_decibel_statistics
from tacotron.params.dataset import dataset_params

if __name__ == '__main__':
    dataset = dataset_params.dataset_loader(dataset_folder=dataset_params.dataset_folder,
                                            char_dict=dataset_params.vocabulary_dict,
                                            fill_dict=True)

    print("Dataset: {}".format(dataset_params.dataset_folder))
    print("Loading dataset ...")
    ids, lens, paths = dataset.load()

    # Collect decibel statistics for all the files.
    print("Collecting decibel statistics for {} files ...".format(len(paths)))
    min_linear_db, max_linear_db, min_mel_db, max_mel_db = collect_decibel_statistics(paths)

    print("avg. max. mel magnitude (dB): mel_mag_ref_db = ", max_mel_db)
    print("avg. min. mel magnitude (dB): mel_mag_max_db = ", min_mel_db)
    print("avg. max. linear magnitude (dB): linear_ref_db = ", max_linear_db)
    print("avg. min. linear magnitude (dB): linear_mag_max_db = ", min_linear_db)
