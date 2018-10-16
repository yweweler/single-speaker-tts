from datasets.statistics import collect_decibel_statistics
from tacotron.params.dataset import dataset_params

if __name__ == '__main__':
    dataset = dataset_params.dataset_loader(dataset_folder=dataset_params.dataset_folder,
                                            char_dict=dataset_params.vocabulary_dict,
                                            fill_dict=True)

    print("Dataset: {}".format(dataset_params.dataset_folder))
    print("Loading dataset ...")
    ids, lens, paths = dataset.load()

    # Pre-compute features for all the files.
    print("Pre-computing features for {} files ...".format(len(paths)))
    dataset.pre_compute_features(paths)