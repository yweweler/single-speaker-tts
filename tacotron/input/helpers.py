import os
from multiprocessing.pool import ThreadPool

import numpy as np

from audio.conversion import inv_normalize_decibel, decibel_to_magnitude, ms_to_samples
from audio.synthesis import spectrogram_to_wav
from tacotron.params.dataset import dataset_params
from tacotron.params.inference import inference_params
from tacotron.params.model import model_params


def py_load_processed_features(wav_path):
    file_path = os.path.splitext(wav_path.decode())[0]

    # Load features from disk.
    data = np.load('{}.npz'.format(file_path))

    return data['mel_mag_db'], data['linear_mag_db']


def __py_pad_sentence(_sentence, _max_len):
    pad_len = _max_len - len(_sentence)
    pad_token = dataset_params.vocabulary_dict['pad']
    _sentence = np.append(_sentence, [pad_token] * pad_len)

    return _sentence


def py_pre_process_sentences(_sentences, dataset):
    # Pre-process sentence and convert it into ids.
    id_sequences, sequence_lengths = dataset.process_sentences(_sentences)

    # Get the first sentence.
    sentences = [np.frombuffer(id_sequence, dtype=np.int32) for id_sequence in id_sequences]

    # Pad sentence to the same length in order to be able to batch them in a single tensor.
    max_length = max(sequence_lengths)
    sentences = np.array([__py_pad_sentence(sentence, max_length) for sentence in sentences])

    print('sentences', sentences)
    print('sentences.shape', sentences.shape)

    return sentences


def __py_synthesize(linear_mag):
    # linear_mag = np.squeeze(linear_mag, -1)
    linear_mag = np.power(linear_mag, model_params.magnitude_power)

    win_len = ms_to_samples(model_params.win_len, model_params.sampling_rate)
    win_hop = ms_to_samples(model_params.win_hop, model_params.sampling_rate)
    n_fft = model_params.n_fft

    print('Spectrogram inversion ...')
    return spectrogram_to_wav(linear_mag,
                              win_len,
                              win_hop,
                              n_fft,
                              model_params.reconstruction_iterations)


def py_post_process_spectrograms(_spectrograms):
    # Apply Griffin-Lim to all spectrogram's to get the waveforms.
    normalized = list()
    for spectrogram in _spectrograms:
        print('Reverse spectrogram normalization ...', spectrogram.shape)
        linear_mag_db = inv_normalize_decibel(spectrogram.T,
                                              dataset_params.dataset_loader.mel_mag_ref_db,
                                              dataset_params.dataset_loader.mel_mag_max_db)

        linear_mag = decibel_to_magnitude(linear_mag_db)
        normalized.append(linear_mag)

    specs = normalized

    # Synthesize waveforms from the spectrograms.
    pool = ThreadPool(inference_params.n_synthesis_threads)
    wavs = pool.map(__py_synthesize, specs)
    pool.close()
    pool.join()

    # # Write all generated waveforms to disk.
    # for i, (sentence, wav) in enumerate(zip(raw_sentences, wavs)):
    #     # Append ".wav" to the sentence line number to get the filename.
    #     file_name = '{}.wav'.format(i + 1)
    #
    #     # Generate the full path under which to save the wav.
    #     save_path = os.path.join(inference_params.synthesis_dir, file_name)
    #
    #     # Write the wav to disk.
    #     # save_wav(save_path, wav, model_params.sampling_rate, True)
    #     print('Saved: "{}"'.format(save_path))

    return wavs


def derive_bucket_boundaries(dataset_loader, n_buckets):
    element_lengths = [row[1] for row in dataset_loader.get_train_listing_generator()]

    # Get the total number of samples in the dataset.
    n_samples = len(element_lengths)

    # Sort sequence lengths in order to slice them into buckets that contain sequences of roughly
    # equal length.
    sorted_sentence_lengths = np.sort(element_lengths)

    if n_samples < n_buckets:
        raise AssertionError('The number of entries loaded is smaller than the number of '
                             'buckets to be created. Automatic calculation of the bucket '
                             'boundaries is not possible.')

    # Slice the sorted lengths into equidistant sections and use the first element of a slice as
    # the bucket boundary.
    bucket_step = n_samples // n_buckets
    bucket_boundaries = sorted_sentence_lengths[::bucket_step]

    # Throw away the first and last bucket boundaries since the bucketing algorithm automatically
    # adds two surrounding ones.
    bucket_boundaries = bucket_boundaries[1:-1].tolist()

    # Remove duplicate boundaries from the list.
    bucket_boundaries = sorted(list(set(bucket_boundaries)))

    print('bucket_boundaries', bucket_boundaries)
    print('n_buckets: {} + 2'.format(len(bucket_boundaries)))

    return bucket_boundaries


def placeholders_from_dataset_iter(dataset_iter):
    ph_sentences, ph_sentence_lengths, ph_mel_specs, ph_lin_specs, ph_time_frames = \
        dataset_iter.get_next()

    # TODO: Technically these are no longer `tf.placeholder` objects.
    # Create batched placeholders from the dataset.
    placeholders = {
        'ph_sentences': ph_sentences,
        'ph_sentence_length': ph_sentence_lengths,
        'ph_mel_specs': ph_mel_specs,
        'ph_lin_specs': ph_lin_specs,
        'ph_time_frames': ph_time_frames
    }

    return placeholders
