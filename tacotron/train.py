import os
import tensorflow as tf
import numpy as np

from datasets.dataset_helper import DatasetHelper
from tacotron.model import Tacotron, Mode
from tacotron.params.dataset import dataset_params
from tacotron.params.model import model_params
from tacotron.params.training import training_params
from tacotron.input.functions import train_input_fn
import sys

# Hack to force tensorflow to run on the CPU.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def batched_placeholders(dataset, max_samples, n_epochs, batch_size):
    """
    Created batches from an dataset that are bucketed by the input sentences sequence lengths.
    Creates placeholders that are filled by QueueRunners. Before executing the placeholder it is
    therefore required to start the corresponding threads using `tf.train.start_queue_runners`.

    Arguments:
        dataset (datasets.DatasetHelper):
            A dataset loading helper that handles loading the data.

        max_samples (int):
            Maximal number of samples to load from the train dataset. If None, all samples from
            the dataset will be used.

        n_epochs (int):
            Number of epochs to train the dataset.

        batch_size (int):
            target size of the batches to create.

    Returns:
        (placeholder_dictionary, n_samples):
            placeholder_dictionary:
                The placeholder dictionary contains the following fields with keys of the same name:
                    - ph_sentences (tf.Tensor):
                        Batched integer sentence sequence with appended <EOS> token padded to same
                        length using the <PAD> token. The characters were converted
                        converted to their vocabulary id's. The shape is shape=(B, T_sent, ?),
                        with B being the batch size and T_sent being the sentence length
                        including the <EOS> token.
                    - ph_sentence_length (tf.Tensor):
                        Batched sequence lengths including the <EOS> token, excluding the padding.
                        The shape is shape=(B), with B being the batch size.
                    - ph_mel_specs (tf.Tensor):
                        Batched Mel. spectrogram's that were padded to the same length in the
                        time axis using zero frames. The shape is shape=(B, T_spec, n_mels),
                        with B being the batch size and T_spec being the number of frames in the
                        spectrogram.
                    - ph_lin_specs (tf.Tensor):
                        Batched linear spectrogram's that were padded to the same length in the
                        time axis using zero frames. The shape is shape=(B, T_spec, 1 + n_fft // 2),
                        with B being the batch size and T_spec being the number of frames in the
                        spectrogram.
                    - ph_time_frames (tf.Tensor):
                        Batched number of frames in the spectrogram's excluding the padding
                        frames. The shape is shape=(B), with B being the batch size.

            n_samples (int):
                Number of samples loaded from the database. Each epoch will contain `n_samples`.
    """
    n_threads = training_params.n_threads

    # Load all sentences and the corresponding audio file paths.
    sentences, sentence_lengths, wav_paths = dataset.load(max_samples=max_samples)
    print('Loaded {} dataset sentences.'.format(len(sentences)))

    # Pre-cache all audio features.
    feature_cache = None
    if training_params.cache_preprocessed:
        feature_cache = DatasetHelper.cache_precalculated_features(wav_paths)
        print('Cached {} waveforms.'.format(len(wav_paths)))

    # Get the total number of samples in the dataset.
    n_samples = len(sentence_lengths)
    print('Finished loading {} dataset entries.'.format(n_samples))

    # Sort sequence lengths in order to slice them into buckets that contain sequences of roughly
    # equal length.
    sorted_sentence_lengths = np.sort(sentence_lengths)

    if n_samples < training_params.n_buckets:
        raise AssertionError('The number of entries loaded is smaller than the number of '
                             'buckets to be created. Automatic calculation of the bucket '
                             'boundaries is not possible.')

    # Slice the sorted lengths into equidistant sections and use the first element of a slice as
    # the bucket boundary.
    bucket_step = n_samples // training_params.n_buckets
    bucket_boundaries = sorted_sentence_lengths[::bucket_step]

    # Throw away the first and last bucket boundaries since the bucketing algorithm automatically
    # adds two surrounding ones.
    bucket_boundaries = bucket_boundaries[1:-1].tolist()

    # Remove duplicate boundaries from the list.
    bucket_boundaries = sorted(list(set(bucket_boundaries)))

    print('bucket_boundaries', bucket_boundaries)
    print('n_buckets: {} + 2'.format(len(bucket_boundaries)))

    # Convert everything into tf.Tensor objects for queue based processing.
    sentences = tf.convert_to_tensor(sentences)
    sentence_lengths = tf.convert_to_tensor(sentence_lengths)
    wav_paths = tf.convert_to_tensor(wav_paths)

    # Create a queue based iterator that yields tuples to process.
    sentence, sentence_length, wav_path = tf.train.slice_input_producer(
        [sentences, sentence_lengths, wav_paths],
        capacity=n_threads * batch_size,
        num_epochs=n_epochs,
        shuffle=training_params.shuffle_samples)

    # The sentence is a integer sequence (char2idx), we need to interpret it as such since it is
    # stored in a tensor that hold objects in order to manage sequences of different lengths in a
    # single tensor.
    sentence = tf.decode_raw(sentence, tf.int32)

    if training_params.load_preprocessed:
        # The training files are expected to be preprocessed to they can be used directly.

        def _load_processed(wav_path):
            file_path = os.path.splitext(wav_path.decode())[0]

            # Either load features from the cache or from disk.
            if training_params.cache_preprocessed:
                data = feature_cache[file_path]
            else:
                data = np.load('{}.npz'.format(file_path))

            return data['mel_mag_db'], data['linear_mag_db']

        mel_spec, lin_spec = tf.py_func(_load_processed, [wav_path], [tf.float32, tf.float32])
    else:
        # Load and process audio file from disk.
        # Apply load_audio to each wav_path of the tensorflow iterator.
        mel_spec, lin_spec = tf.py_func(dataset.load_audio, [wav_path], [tf.float32, tf.float32])

    # The shape of the returned values from py_func seems to get lost for some reason.
    mel_spec.set_shape((None, model_params.n_mels * model_params.reduction))
    lin_spec.set_shape((None, (1 + model_params.n_fft // 2) * model_params.reduction))

    # Get the number spectrogram time-steps (used as the number of time frames when generating).
    n_time_frames = tf.shape(mel_spec)[0]

    # Determine the bucket capacity for each bucket.
    bucket_capacities = [training_params.n_samples_per_bucket] * (len(bucket_boundaries) + 1)

    # Batch data based on sequence lengths.
    ph_sentence_length, (ph_sentences, ph_mel_specs, ph_lin_specs, ph_time_frames) = \
        tf.contrib.training.bucket_by_sequence_length(
            input_length=sentence_length,
            tensors=[sentence, mel_spec, lin_spec, n_time_frames],
            batch_size=batch_size,
            bucket_boundaries=bucket_boundaries,
            num_threads=n_threads,
            capacity=training_params.n_pre_calc_batches,
            bucket_capacities=bucket_capacities,
            dynamic_pad=True,
            allow_smaller_final_batch=training_params.allow_smaller_batches)

    # print('batched.ph_sentence_length', ph_sentence_length.shape, ph_sentence_length)
    # print('batched.ph_sentences.shape', ph_sentences.shape, ph_sentences)
    # print('batched.ph_mel_specs.shape', ph_mel_specs.shape, ph_mel_specs)
    # print('batched.ph_lin_specs.shape', ph_lin_specs.shape, ph_lin_specs)
    # print('batched.ph_time_frames', ph_time_frames.shape, ph_time_frames)

    # Collect all created placeholder in a dictionary.
    placeholder_dict = {
        'ph_sentences': ph_sentences,
        'ph_sentence_length': ph_sentence_length,
        'ph_mel_specs': ph_mel_specs,
        'ph_lin_specs': ph_lin_specs,
        'ph_time_frames': ph_time_frames
    }

    return placeholder_dict, n_samples


def train(model):
    """
    Trains a Tacotron model.

    Arguments:
        model (Tacotron):
            The Tacotron model instance to be trained.
    """
    # NOTE: The global step has to be created before the optimizer is created.
    global_step = tf.train.create_global_step()

    # Get the models loss function.
    loss_op = model.get_loss_op()

    with tf.name_scope('optimizer'):
        # Let the learning rate decay exponentially.
        learning_rate = tf.train.exponential_decay(
            learning_rate=training_params.lr,
            global_step=global_step,
            decay_steps=training_params.lr_decay_steps,
            decay_rate=training_params.lr_decay_rate,
            staircase=training_params.lr_staircase)

        # Force decrease to stop at a minimal learning rate.
        learning_rate = tf.maximum(learning_rate, training_params.minimum_lr)

        # Add a learning rate summary.
        tf.summary.scalar('lr', learning_rate)

        # Create a optimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Apply gradient clipping by global norm.
        gradients, variables = zip(*optimizer.compute_gradients(loss_op))
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, training_params.gradient_clip_norm)

        # Add dependency on UPDATE_OPS; otherwise batch normalization won't work correctly.
        # See: https://github.com/tensorflow/tensorflow/issues/1122
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimize = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step)

    # Create the training session.
    session = start_session(loss_op=loss_op, summary_op=model.summary())

    # Start training.
    while not session.should_stop():
        try:
            session.run([global_step, loss_op, optimize])
        except tf.errors.OutOfRangeError:
            break

    session.close()


def start_session(loss_op, summary_op):
    """
    Creates a session that can be used for training.

    Arguments:
        loss_op (tf.Tensor):

        summary_op (tf.Tensor):
            A tensor of type `string` containing the serialized `Summary` protocol
            buffer containing all merged model summaries.

    Returns:
        tf.train.SingularMonitoredSession
    """
    checkpoint_dir = os.path.join(training_params.checkpoint_dir, training_params.checkpoint_run)

    saver = tf.train.Saver(
            # NOTE: CUDNN RNNs do not support distributed saving of parameters.
            sharded=False,
            allow_empty=True,
            max_to_keep=training_params.checkpoints_to_keep,
            save_relative_paths=True
        )

    saver_hook = tf.train.CheckpointSaverHook(
        checkpoint_dir=checkpoint_dir,
        # save_secs=training_params.checkpoint_save_secs,
        save_steps=training_params.checkpoint_save_steps,
        saver=saver
    )

    summary_hook = tf.train.SummarySaverHook(
        output_dir=checkpoint_dir,
        save_steps=training_params.summary_save_steps,
        summary_op=summary_op
    )

    nan_hook = tf.train.NanTensorHook(
        loss_tensor=loss_op,
        fail_on_nan_loss=True
    )

    counter_hook = tf.train.StepCounterHook(
        output_dir=checkpoint_dir,
        every_n_steps=training_params.performance_log_steps
    )

    session_config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
        )
    )

    session = tf.train.SingularMonitoredSession(hooks=[
        saver_hook,
        summary_hook,
        nan_hook,
        counter_hook],
        # Note: When using a monitored session in combination with CUDNN RNNs this needs to be
        # set otherwise the CUDNN RNN does not find a default device to collect variables for
        # saving.
        scaffold=tf.train.Scaffold(saver=saver),
        config=session_config,
        checkpoint_dir=checkpoint_dir)

    tf.train.start_queue_runners(sess=session)

    return session


def main(_):
    # Create a dataset loader.
    train_dataset = dataset_params.dataset_loader(dataset_folder=dataset_params.dataset_folder,
                                                  char_dict=dataset_params.vocabulary_dict,
                                                  fill_dict=False)

    dataset_iter = train_input_fn(
        dataset_loader=train_dataset,
        max_samples=training_params.max_samples
    )

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

    # Create the Tacotron model.
    tacotron_model = Tacotron(inputs=placeholders, mode=Mode.TRAIN,
                              training_summary=training_params.write_summary)

    # Train the model.
    train(tacotron_model)


# def main(_):
#     # Create a dataset loader.
#     train_dataset = dataset_params.dataset_loader(dataset_folder=dataset_params.dataset_folder,
#                                                   char_dict=dataset_params.vocabulary_dict,
#                                                   fill_dict=False)
#
#     # tf.enable_eager_execution()
#
#     next_element = train_input_fn(
#         dataset_loader=train_dataset,
#         max_samples=training_params.max_samples
#     )
#     print('next_element', next_element)
#
#     batch_iter = next_element.get_next()
#
#     with tf.Session() as session:
#         for i in range(5):
#             batch = session.run(batch_iter)
#             print('Dataset batch elements:', len(batch))
#             for elem_id, e in enumerate(batch):
#                 print('batch.elem[{}]: type={}, shape={}, data={}'
#                       .format(elem_id, type(e), e.shape, None))
#             print('======================')
#
#     print('The End.')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
