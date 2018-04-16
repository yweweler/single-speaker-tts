import os

import tensorflow as tf

from tacotron.model import Tacotron
from tacotron.params.dataset import dataset_params
from tacotron.params.evaluation import evaluation_params
from tacotron.params.model import model_params

# Hack to force tensorflow to run on the CPU.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

tf.logging.set_verbosity(tf.logging.INFO)


def batched_placeholders(dataset, max_samples, batch_size):
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
    n_threads = evaluation_params.n_threads

    # Load alĺ sentences and the corresponding audio file paths.
    sentences, sentence_lengths, wav_paths = dataset.load(max_samples=max_samples)

    # Determine the minimal and maximal sentence lengths for calculating the bucket boundaries.
    max_len, min_len = max(sentence_lengths), min(sentence_lengths)

    # Get the total number of samples in the dataset.
    n_samples = len(sentence_lengths)

    # Convert everything into tf.Tensor objects for queue based processing.
    sentences = tf.convert_to_tensor(sentences)
    sentence_lengths = tf.convert_to_tensor(sentence_lengths)
    wav_paths = tf.convert_to_tensor(wav_paths)

    # Create a queue based iterator that yields tuples to process.
    sentence, sentence_length, wav_path = tf.train.slice_input_producer(
        [sentences, sentence_lengths, wav_paths],
        capacity=n_threads * batch_size,
        num_epochs=1,
        shuffle=evaluation_params.shuffle_samples)

    # The sentence is a integer sequence (char2idx), we need to interpret it as such since it is
    # stored in a tensor that hold objects in order to manage sequences of different lengths in a
    # single tensor.
    sentence = tf.decode_raw(sentence, tf.int32)

    # Apply load_audio to each wav_path of the tensorflow iterator.
    mel_spec, lin_spec = tf.py_func(dataset.load_audio, [wav_path], [tf.float32, tf.float32])

    # The shape of the returned values from py_func seems to get lost for some reason.
    mel_spec.set_shape((None, model_params.n_mels))
    lin_spec.set_shape([None, (1 + model_params.n_fft // 2)])

    # Get the number spectrogram time-steps (used as the number of time frames when generating).
    n_time_frames = tf.shape(mel_spec)[0]

    # TODO: Calculate better bucket boundaries. (See: #8)
    buckets = [i for i in range(min_len + 1, max_len + 1, 8)]
    print('n_buckets: {} + 2'.format(len(buckets)))

    # Batch data based on sequence lengths.
    ph_sentence_length, (ph_sentences, ph_mel_specs, ph_lin_specs, ph_time_frames) = \
        tf.contrib.training.bucket_by_sequence_length(
            input_length=sentence_length,
            tensors=[sentence, mel_spec, lin_spec, n_time_frames],
            batch_size=batch_size,
            bucket_boundaries=buckets,
            num_threads=n_threads,
            capacity=evaluation_params.n_pre_calc_batches,
            bucket_capacities=[evaluation_params.n_samples_per_bucket] * (len(buckets) + 1),
            dynamic_pad=True,
            allow_smaller_final_batch=evaluation_params.allow_smaller_batches)

    # Collect all created placeholder in a dictionary.
    placeholder_dict = {
        'ph_sentences': ph_sentences,
        'ph_sentence_length': ph_sentence_length,
        'ph_mel_specs': ph_mel_specs,
        'ph_lin_specs': ph_lin_specs,
        'ph_time_frames': ph_time_frames
    }

    return placeholder_dict, n_samples


def evaluate(model, n_samples):
    """
    Evaluates a Tacotron model.

    Arguments:
        model (Tacotron):
            The Tacotron model instance to be evaluated.

        n_samples (int):
            Number of samples used for evaluation.
    """
    # Get the models loss function.
    loss_op = model.get_loss_op()

    # Checkpoint folder to load the evaluation checkpoint from.
    checkpoint_load_dir = os.path.join(
        evaluation_params.checkpoint_dir,
        evaluation_params.checkpoint_load_run
    )

    checkpoint_file = tf.train.latest_checkpoint(checkpoint_load_dir)

    # Checkpoint folder to save the evaluation summaries into.
    checkpoint_save_dir = os.path.join(
        evaluation_params.checkpoint_dir,
        evaluation_params.checkpoint_save_run
    )

    saver = tf.train.Saver()

    summary_writer = tf.summary.FileWriter(checkpoint_save_dir, tf.get_default_graph())
    summary_op = tf.summary.merge_all()

    # Create the evaluation session.
    session = start_session()

    print('Restoring model...')
    saver.restore(session, checkpoint_file)
    print('Restoring finished')

    loss_sum = 0

    # Start evaluation.
    while True:
        try:
            loss = session.run([loss_op])
            # Accumulate the batch losses.
            loss_sum += loss[0]
            print(loss[0])
        except tf.errors.OutOfRangeError:
            break

    # Calculate the average batch loss.
    avg_loss = loss_sum / n_samples
    print(avg_loss)

    # Collect the models summaries and add the evaluation loss.
    eval_summary = tf.Summary()
    eval_summary.ParseFromString(session.run(summary_op))
    eval_summary.value.add(tag='loss/loss', simple_value=avg_loss)

    summary_writer.add_summary(eval_summary, 1)

    session.close()


def start_session():
    """
    Creates a session that can be used for training.

    Returns:
        tf.Session
    """

    session_config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
        )
    )

    session = tf.Session(config=session_config)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    session.run(init_op)

    tf.train.start_queue_runners(sess=session)

    return session


if __name__ == '__main__':
    # Create a dataset loader.
    train_dataset = dataset_params.dataset_loader(dataset_folder=dataset_params.dataset_folder,
                                                  char_dict=dataset_params.vocabulary_dict,
                                                  fill_dict=False)

    # Create batched placeholders from the dataset.
    placeholders, n_samples = batched_placeholders(dataset=train_dataset,
                                                   max_samples=evaluation_params.max_samples,
                                                   batch_size=evaluation_params.batch_size)

    # Create the Tacotron model.
    tacotron_model = Tacotron(inputs=placeholders, training=False)

    # Train the model.
    evaluate(tacotron_model, n_samples)
