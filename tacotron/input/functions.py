import tensorflow as tf
from tensorflow.python.data.experimental.ops import grouping
import tensorflow.python.data.experimental as tf_experimental

import numpy as np

from tacotron.input.helpers import derive_bucket_boundaries, py_pre_process_sentences, \
    py_load_processed_features
from tacotron.params.evaluation import evaluation_params
from tacotron.params.model import model_params
from tacotron.params.training import training_params
from tacotron.params.inference import inference_params


def train_input_fn(dataset_loader):
    return __build_input_fn(dataset_loader=dataset_loader,
                            max_samples=training_params.max_samples,
                            batch_size=training_params.batch_size,
                            n_epochs=training_params.n_epochs,
                            n_threads=training_params.n_threads,
                            cache_preprocessed=training_params.cache_preprocessed,
                            load_preprocessed=training_params.load_preprocessed,
                            shuffle_samples=training_params.shuffle_samples,
                            n_buckets=training_params.n_buckets,
                            n_pre_calc_batches=training_params.n_pre_calc_batches,
                            model_n_mels=model_params.n_mels,
                            model_reduction=model_params.reduction,
                            model_n_fft=model_params.n_fft)


def eval_input_fn(dataset_loader):
    return __build_input_fn(dataset_loader=dataset_loader,
                            max_samples=evaluation_params.max_samples,
                            batch_size=evaluation_params.batch_size,
                            n_epochs=1,
                            n_threads=evaluation_params.n_threads,
                            cache_preprocessed=False,
                            load_preprocessed=False,
                            shuffle_samples=evaluation_params.shuffle_samples,
                            n_buckets=evaluation_params.n_buckets,
                            n_pre_calc_batches=evaluation_params.n_pre_calc_batches,
                            model_n_mels=model_params.n_mels,
                            model_reduction=model_params.reduction,
                            model_n_fft=model_params.n_fft)


def inference_input_fn(dataset_loader, sentence_generator):
    return __build_inference_input_fn(dataset_loader=dataset_loader,
                                      sentence_generator=sentence_generator,
                                      n_threads=inference_params.n_synthesis_threads)


def __build_inference_input_fn(dataset_loader, sentence_generator, n_threads):
    def __input_fn():
        dataset = tf.data.Dataset.from_generator(sentence_generator,
                                                 (tf.int32),
                                                 (tf.TensorShape([None, ])))

        dataset = dataset.batch(1)

        # Create an iterator over the dataset.
        iterator = dataset.make_one_shot_iterator()

        # Get features from the iterator.
        ph_sentences = iterator.get_next()

        features = {
            'ph_sentences': ph_sentences
        }

        return features, None

    return __input_fn


def __build_input_fn(dataset_loader, max_samples, batch_size, n_epochs, n_threads,
                     cache_preprocessed, load_preprocessed,
                     shuffle_samples, n_buckets, n_pre_calc_batches, model_n_mels, model_reduction,
                     model_n_fft):
    def __input_fn():
        # TODO: Add flag: consumption of the generator and alternative use of `from_tensor_slices`.
        # dataset = tf.data.Dataset.from_tensor_slices((sentences, sentence_lengths, wav_paths))
        # dataset_generator = dataset_loader.get_train_listing_generator(max_samples=max_samples)
        dataset_generator = dataset_loader.get_train_listing_generator()

        # TODO: Rewrite so that this also works for evaluation.
        def _generator():
            for _element in dataset_generator:
                yield _element['tokenized_sentence'], \
                      _element['tokenized_sentence_length'], \
                      _element['audio_path']

        dataset = tf.data.Dataset.from_generator(
            _generator,
            (tf.int32, tf.int32, tf.string),
            (tf.TensorShape([None, ]), tf.TensorShape([]), tf.TensorShape([]))
        )
        #    args=[max_samples])

        def __element_pre_process_fn(sentence, sentence_length, wav_path):
            # TODO: Rewrite this to use tensorflow functions only.
            if load_preprocessed:
                # Load pre-calculated features from disk.
                mel_spec, lin_spec = tf.py_func(py_load_processed_features,
                                                [wav_path],
                                                [tf.float32, tf.float32])
            else:
                # Calculate features on the fly.
                mel_spec, lin_spec = tf.py_func(dataset_loader.load_audio,
                                                [wav_path],
                                                [tf.float32, tf.float32])

            # The shape of the returned values from py_func seems to get lost for some reason.
            mel_spec.set_shape((None, model_n_mels * model_reduction))
            lin_spec.set_shape((None, (1 + model_n_fft // 2) * model_reduction))

            # print_op = tf.print("sentence_length:", sentence_length, output_stream=sys.stdout)
            # with tf.control_dependencies([print_op]):
            #     mel_spec = tf.identity(mel_spec)

            # Get the number spectrogram time-steps (number of time frames when generating).
            n_time_frames = tf.shape(mel_spec)[0]

            processed_tensors = (
                sentence,
                # tf.decode_raw(sentence, tf.int32),
                sentence_length,
                mel_spec,
                lin_spec,
                n_time_frames
            )
            return processed_tensors

        # Pre-process dataset elements.
        print('Applying pre-processing ...')
        dataset = dataset.map(__element_pre_process_fn, num_parallel_calls=n_threads)

        # TODO: Input pipeline statistics are not working right now.
        # I can not get the aggregator to collect anything from the graph nor can I make it
        # produce summaries that can be shown in tensorboard.
        # stats_aggregator = tf_experimental.StatsAggregator()
        # dataset = dataset.apply(tf_experimental.set_stats_aggregator(stats_aggregator))
        # dataset = dataset.apply(tf_experimental.latency_stats("total_bytes"))
        # stats_summary = stats_aggregator.get_summary()
        # tf.add_to_collection(tf.GraphKeys.SUMMARIES, stats_summary)

        # Feature caching.
        if cache_preprocessed:
            # Cache dataset elements (including the calculated features) in RAM.
            dataset = dataset.cache()

        # Repeat epochs and shuffle.
        if shuffle_samples:
            # TODO: Rework the hyper-params to enable setting this manually.
            # buffer_size: the maximum number elements that will be buffered when pre-fetching.
            buffer_size = batch_size * n_threads

            # Repeat dataset for the requested number of epochs and shuffle the cache with each epoch.
            dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size, n_epochs))
        else:
            # Repeat dataset for the requested number of epochs without shuffle.
            dataset = dataset.repeat(n_epochs)

        def __element_length_fn(sentence, sentence_length, mel_spec, lin_spec, n_time_frames):
            del sentence
            del mel_spec
            del lin_spec
            del n_time_frames

            return sentence_length

        # TODO: Rewrite so that this also works for evaluation.
        # Derive the bucket boundaries based on the distribution of all sequence lengths in the
        # training portion of the dataset.
        print('Deriving bucket boundaries ...')
        bucket_boundaries = derive_bucket_boundaries(dataset_loader, n_buckets)
        print('Derived boundaries: {}'.format(bucket_boundaries))

        # Use the same batch_size for all buckets.
        bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)

        # TODO: Wait for PR to enable bucketing to provide full batches only.
        # https://github.com/tensorflow/tensorflow/pull/24071
        print('Applying bucketing ...')
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

        # Prefetch batches.
        dataset = dataset.prefetch(n_pre_calc_batches)

        # Create an iterator over the dataset.
        # iterator = dataset.make_one_shot_iterator()
        iterator = dataset.make_initializable_iterator()

        # TODO: Not sure if this is the recommended way to go.
        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

        # Get features from the iterator.
        ph_sentences, ph_sentence_lengths, ph_mel_specs, ph_lin_specs, ph_time_frames =\
            iterator.get_next()

        features = {
            'ph_sentences': ph_sentences,
            'ph_sentence_lengths': ph_sentence_lengths,
            'ph_mel_specs': ph_mel_specs,
            'ph_lin_specs': ph_lin_specs,
            'ph_time_frames': ph_time_frames
        }

        for k, f in features.items():
            print(k, f)

        return features, None

    return __input_fn
