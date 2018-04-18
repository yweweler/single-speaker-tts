import numpy as np
import pytest

from datasets.lj_speech import LJSpeechDatasetHelper

DATASET_PATH = '/home/yves-noel/downloads/LJSpeech-1.1'

INIT_CHAR_DICT = vocabulary_dict = {
    'pad': 0,  # padding
    'eos': 1,  # end of sequence
}


@pytest.fixture
def dataset():
    """
    Creates a LJSpeechDatasetHelper instance.

    Returns:
        LJSpeechDatasetHelper
    """
    _dataset = LJSpeechDatasetHelper(dataset_folder=DATASET_PATH,
                                     char_dict=INIT_CHAR_DICT,
                                     fill_dict=True)

    return _dataset


def test_load_all(dataset):
    """
    Test loading all samples from the dataset.

    Args:
        dataset (LJSpeechDatasetHelper):
            Dataset loading helper instance.
    """

    # The total number of samples in the dataset.
    n_samples = 13100

    id_sentences, sentence_lengths, file_paths = dataset.load(max_samples=None,
                                                              min_len=None,
                                                              max_len=None)

    assert len(id_sentences) == n_samples
    assert len(sentence_lengths) == n_samples
    assert len(file_paths) == n_samples


def test_load_n(dataset):
    """
    Test loading only N samples from the dataset.

    Args:
        dataset (LJSpeechDatasetHelper):
            Dataset loading helper instance.
    """

    # The total number of samples to load.
    n_samples = 3438

    id_sentences, sentence_lengths, file_paths = dataset.load(max_samples=n_samples,
                                                              min_len=None,
                                                              max_len=None)

    assert len(id_sentences) == n_samples
    assert len(sentence_lengths) == n_samples
    assert len(file_paths) == n_samples


def test_load_n_in_range(dataset):
    """
    Test loading only N samples from the dataset with a sentence length in a certain range.

    Args:
        dataset (LJSpeechDatasetHelper):
            Dataset loading helper instance.
    """

    # The total number of samples to load.
    n_samples = 456
    min_len = 13
    max_len = 98

    id_sentences, sentence_lengths, file_paths = dataset.load(max_samples=n_samples,
                                                              min_len=min_len,
                                                              max_len=max_len)

    # Assert the number of loaded samples.
    assert len(id_sentences) == n_samples
    assert len(sentence_lengths) == n_samples
    assert len(file_paths) == n_samples

    # Loaded sentences are folded and processed their loaded length does not longer match the
    # requested length range. The range is only checked against the non folded sentences.


def test_load_sentence_folding(dataset):
    """
    Test folding (replace unwanted characters / lowercase conversion) of sentences.

    Args:
        dataset (LJSpeechDatasetHelper):
            Dataset loading helper instance.
    """
    # First sentence from metadata.csv.
    test_sentence = 'Printing, in the only sense with which we are at present concerned, differs ' \
                    'from most if not from all the arts and crafts represented in the Exhibition'

    folded_sentence = 'printing in the only sense with which we are at present concerned differs ' \
                      'from most if not from all the arts and crafts represented in the exhibition'

    sentence = dataset.replace_abbreviations(test_sentence.lower())

    assert folded_sentence == sentence


def test_load_id_conversion(dataset):
    """
    Test conversion of sentences to ids and back.

    Args:
        dataset (LJSpeechDatasetHelper):
            Dataset loading helper instance.
    """

    # Load all samples from the dataset.
    n_samples = None

    id_sentences, sentence_lengths, file_paths = dataset.load(max_samples=n_samples,
                                                              min_len=None,
                                                              max_len=None)

    # ==============================================================================================
    # (Manually folded) 1. sentence from metadata.csv.
    # ==============================================================================================
    pos = 1 - 1
    folded_sentence = 'printing in the only sense with which we are at present concerned differs ' \
                      'from most if not from all the arts and crafts represented in the exhibition'

    # First sentence loaded by the loading helper.
    # [:-1] to remove the EOS token.
    sentence = np.fromstring(id_sentences[pos], dtype=np.int32)[:-1]

    # Test reconstruction.
    restored_sentence = dataset.idx2sent(sentence)
    assert folded_sentence == restored_sentence

    # ==============================================================================================
    # (Manually folded) 5643. sentence from metadata.csv.
    # ==============================================================================================
    pos = 5643 - 1
    folded_sentence = 'the prisoners were to cultivate the land and raise sufficient produce for ' \
                      'their own support'

    # First sentence loaded by the loading helper.
    # [:-1] to remove the EOS token.
    sentence = np.fromstring(id_sentences[pos], dtype=np.int32)[:-1]

    # Test reconstruction.
    restored_sentence = dataset.idx2sent(sentence)
    assert folded_sentence == restored_sentence


def test_trailing_eos(dataset):
    """
    Test if sentences in id representation always have an trailing EOS token.

    Args:
        dataset (LJSpeechDatasetHelper):
            Dataset loading helper instance.
    """

    # Load all samples from the dataset.
    n_samples = None

    id_sentences, sentence_lengths, file_paths = dataset.load(max_samples=n_samples,
                                                              min_len=None,
                                                              max_len=None)

    # Check for each token if it ends with an EOS token.
    for sentence in id_sentences:
        assert np.fromstring(sentence, dtype=np.int32)[-1] == INIT_CHAR_DICT['eos']
