import tensorflow as tf
from tensorflow.python.data.experimental.ops import grouping

from tacotron.input.helpers import derive_bucket_boundaries
from tacotron.params.model import model_params
from tacotron.params.training import training_params
from tacotron.params.evaluation import evaluation_params


def train_input_fn(dataset_loader):
    return _input_fn(dataset_loader=dataset_loader,
                     max_samples=training_params.max_samples,
                     batch_size=training_params.batch_size,
                     n_epochs=training_params.n_epochs,
                     n_threads=training_params.n_threads,
                     cache_preprocessed=training_params.cache_preprocessed,
                     shuffle_samples=training_params.shuffle_samples,
                     n_buckets=training_params.n_buckets,
                     n_pre_calc_batches=training_params.n_pre_calc_batches,
                     model_n_mels=model_params.n_mels,
                     model_reduction=model_params.reduction,
                     model_n_fft=model_params.n_fft)


def eval_input_fn(dataset_loader):
    return _input_fn(dataset_loader=dataset_loader,
                     max_samples=evaluation_params.max_samples,
                     batch_size=evaluation_params.batch_size,
                     n_epochs=1,
                     n_threads=evaluation_params.n_threads,
                     cache_preprocessed=False,
                     shuffle_samples=evaluation_params.shuffle_samples,
                     n_buckets=evaluation_params.n_buckets,
                     n_pre_calc_batches=evaluation_params.n_pre_calc_batches,
                     model_n_mels=model_params.n_mels,
                     model_reduction=model_params.reduction,
                     model_n_fft=model_params.n_fft)


def _input_fn(dataset_loader, max_samples, batch_size, n_epochs, n_threads, cache_preprocessed,
              shuffle_samples, n_buckets, n_pre_calc_batches, model_n_mels, model_reduction,
              model_n_fft):
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
        mel_spec.set_shape((None, model_n_mels * model_reduction))
        lin_spec.set_shape((None, (1 + model_n_fft // 2) * model_reduction))

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

    # TODO: Implement loading of pre-calculated features.
    # Pre-process dataset elements.
    dataset = dataset.map(__element_pre_process_fn, num_parallel_calls=n_threads)

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

    # Derive the bucket boundaries based on the distribution of all sequence lengths in the dataset.
    bucket_boundaries = derive_bucket_boundaries(sentence_lengths, n_buckets)

    # Use the same batch_size for all buckets.
    bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)

    # TODO: Implement `allow_smaller_batches`.
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

    # Prefetch batches.
    dataset = dataset.prefetch(n_pre_calc_batches)

    # Create an iterator over the dataset.
    iterator = dataset.make_one_shot_iterator()

    return iterator
