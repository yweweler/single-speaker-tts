import numpy as np
import tensorflow as tf
from datasets.dataset_helper import DatasetHelper
from tacotron.params.dataset import dataset_params
from tacotron.params.model import model_params
from tacotron.params.training import training_params
from tensorflow.python.data.experimental.ops import grouping
import sys
from tacotron.input.helpers import derive_bucket_boundaries


def train_input_fn(dataset_loader, max_samples):
    print('entered train_input_fn')
    return _input_fn(dataset_loader, max_samples, n_epochs=training_params.n_epochs)


# TODO: Refactor to make the function more general so train/eval can use it (maybe even predict).
def _input_fn(dataset_loader, max_samples, n_epochs):
    print('entered _input_fn')

    # Load all sentences and the corresponding audio file paths.
    sentences, sentence_lengths, wav_paths = dataset_loader.load(max_samples=max_samples)
    print('Loaded {} dataset sentences.'.format(len(sentences)))

    # TODO: Compare the performance with `tf.data.Dataset.from_generator`.
    dataset = tf.data.Dataset.from_tensor_slices((sentences, sentence_lengths, wav_paths))

    def __element_pre_process_fn(sentence, sentence_length, wav_path):
        # TODO: Rewrite this to use tensorflow functions only.
        mel_spec, lin_spec = tf.py_func(dataset_loader.load_audio,
                                        [wav_path],
                                        [tf.float32, tf.float32])

        # The shape of the returned values from py_func seems to get lost for some reason.
        mel_spec.set_shape((None, model_params.n_mels * model_params.reduction))
        lin_spec.set_shape((None, (1 + model_params.n_fft // 2) * model_params.reduction))

        # print_op = tf.print("sentence_length:", sentence_length, output_stream=sys.stdout)
        # with tf.control_dependencies([print_op]):
        #     mel_spec = tf.identity(mel_spec)

        # Get the number spectrogram time-steps (used as the number of time frames when generating).
        n_time_frames = tf.shape(mel_spec)[0]

        processed_tensors = (
            tf.decode_raw(sentence, tf.int32),
            sentence_length,
            mel_spec,
            lin_spec,
            n_time_frames
        )
        return processed_tensors

    dataset = dataset.map(__element_pre_process_fn)
    dataset = dataset.cache()

    def __element_length_fn(sentence, sentence_length, mel_spec, lin_spec, n_time_frames):
        del sentence
        del mel_spec
        del lin_spec
        del n_time_frames
        return sentence_length

    bucket_boundaries = derive_bucket_boundaries(sentence_lengths, training_params.n_buckets)
    bucket_batch_sizes = [training_params.batch_size] * (len(bucket_boundaries) + 1)

    print('Starting bucket collection...')
    # TODO: Open a PR to enable bucket_by_sequence_length to provide full batches only.
    dataset = dataset.apply(
        # Bucket dataset elements based on sequence_length.
        # By default sequences are padded using 0 up to the longest sequence in each batch.
        grouping.bucket_by_sequence_length(
            element_length_func=__element_length_fn,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            pad_to_bucket_boundary=False,
        )
    )

    # TODO: Add new flags to the hyper-params.
    # print('Prefetch...')
    # dataset = dataset.prefetch(5)
    dataset = dataset.repeat(n_epochs)

    iterator = dataset.make_one_shot_iterator()
    return iterator
    # ds_sentences, ds_sentence_lengths, ds_mel_specs, ds_lin_specs, ds_n_time_frames =\
    #     iterator.get_next()

    # features = {
    #     'ds_sentences': ds_sentences,
    #     'ds_sentence_lengths': ds_sentence_lengths,
    #     'ds_mel_specs': ds_mel_specs,
    #     'ds_lin_specs': ds_lin_specs,
    #     'ds_time_frames': ds_n_time_frames,
    # }

    # return features, None
