import numpy as np
import tensorflow as tf

from audio.conversion import ms_to_samples, magnitude_to_decibel, normalize_decibel
from audio.features import mel_scale_spectrogram, linear_scale_spectrogram
from audio.io import load_wav
from tacotron.hparams import hparams
from tacotron.model import Tacotron

# Hack to force tensorflow to run on the CPU.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

tf.logging.set_verbosity(tf.logging.INFO)


def apply_reduction_padding(mel_mag_db, linear_mag_db, reduction_factor):
    """
    Adds zero padding frames in the time axis to the Mel. scale and linear scale spectrogram's such
    that the number of frames is a multiple of the `reduction_factor`.

    Arguments:
        mel_mag_db (np.ndarray):
            Mel scale spectrogram to be reduced. The shape is expected to be shape=(T_spec, n_mels),
            with T_spec being the number of frames.

        linear_mag_db (np.ndarray):
            Linear scale spectrogram to be reduced. The shape is expected to be
            shape=(T_spec, 1 + n_fft // 2), with T_spec being the number of frames.

        reduction_factor (int):
            The number of consecutive frames.

    Returns:
        (mel_mag_db, linear_mag_db):
            mel_mag_db (np.ndarray):
                Padded Mel. scale spectrogram.
            linear_mag_db (np.ndarray):
                padded linear scale spectrogram.
    """
    # Number of frames in the spectrogram's.
    n_frames = mel_mag_db.shape[0]

    # Make sure the number of frames is a multiple of `reduction_factor` for reduction.
    if (n_frames % reduction_factor) != 0:
        # Calculate the number of padding frames that have to be added.
        n_padding_frames = reduction_factor - (n_frames % reduction_factor)

        # Add padding frames containing zeros to the mel spectrogram.
        mel_mag_db = np.pad(mel_mag_db, [[0, n_padding_frames], [0, 0]], mode="constant")

        # Pad the linear spectrogram since it has to have the same num. of frames.
        linear_mag_db = np.pad(linear_mag_db, [[0, n_padding_frames], [0, 0]], mode="constant")

    # Reduce `reduction_factor` consecutive frames into a single frame.
    # mel_mag_db = mel_mag_db.reshape((-1, mel_mag_db.shape[1] * reduction_factor))
    # linear_mag_db = linear_mag_db.reshape((-1, linear_mag_db.shape[1] * reduction_factor))

    return mel_mag_db, linear_mag_db


def load_audio(file_path):
    """
    Load and pre-process and audio file.

    Arguments:
        file_path (bytes):
            Path to the file to be loaded.

    Returns:
        (mel_mag_db, linear_mag_db):
            mel_mag_db (np.ndarray):
                Mel. scale magnitude spectrogram of the loaded audio file.
                The spectrogram's dB values are normalized to the range [0.0, 1.0] over the
                entire dataset.
                The shape is shape=(T_spec, n_mels), with T_spec being the number of frames in
                the spectrogram.

            linear_mag_db (np.ndarray):
                Linear scale magnitude spectrogram of the loaded audio file.
                The spectrogram's dB values are normalized to the range [0.0, 1.0] over the
                entire dataset.
                The shape is shape=(T_spec, 1 + n_fft // 2), with T_spec being the number of
                frames in the spectrogram.
    """
    # Window length in audio samples.
    win_len = ms_to_samples(hparams.win_len, hparams.sampling_rate)
    # Window hop in audio samples.
    hop_len = ms_to_samples(hparams.win_hop, hparams.sampling_rate)

    # Load the actual audio file.
    wav, sr = load_wav(file_path.decode())

    # TODO: Determine a better silence reference level for the LJSpeech dataset (See: #9).
    # Remove silence at the beginning and end of the wav so the network does not have to learn
    # some random initial silence delay after which it is allowed to speak.
    # wav, _ = trim_silence(wav)

    # Calculate the linear scale spectrogram.
    # Note the spectrogram shape is transposed to be (T_spec, 1 + n_fft // 2) so dense layers for
    # example are applied to each frame automatically.
    linear_spec = linear_scale_spectrogram(wav, hparams.n_fft, hop_len, win_len).T

    # Calculate the Mel. scale spectrogram.
    # Note the spectrogram shape is transposed to be (T_spec, n_mels) so dense layers for example
    # are applied to each frame automatically.
    mel_spec = mel_scale_spectrogram(wav, hparams.n_fft, sr, hparams.n_mels,
                                     hparams.mel_fmin, hparams.mel_fmax, hop_len, win_len, 1).T

    # Convert the linear spectrogram into decibel representation.
    linear_mag = np.abs(linear_spec)
    linear_mag_db = magnitude_to_decibel(linear_mag)
    linear_mag_db = normalize_decibel(linear_mag_db, 35.7, 100)  # TODO: Refactor numbers. (See: #9)
    # => linear_mag_db.shape = (T_spec, 1 + n_fft // 2)

    # Convert the mel spectrogram into decibel representation.
    mel_mag = np.abs(mel_spec)
    mel_mag_db = magnitude_to_decibel(mel_mag)
    mel_mag_db = normalize_decibel(mel_mag_db, 6.0, 100)  # TODO: Refactor numbers. (See: #9)
    # => mel_mag_db.shape = (T_spec, n_mels)

    # Tacotron reduction factor.
    if hparams.reduction > 1:
        mel_mag_db, linear_mag_db = apply_reduction_padding(mel_mag_db, linear_mag_db,
                                                            hparams.reduction)

    # print("[REDUCED] load_audio.mel_spec.shape", np.array(mel_mag_db).astype(np.float32).shape)

    return np.array(mel_mag_db).astype(np.float32), \
           np.array(linear_mag_db).astype(np.float32)


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
    n_threads = hparams.train.n_threads

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
        num_epochs=n_epochs,
        shuffle=hparams.train.shuffle_smaples)

    # The sentence is a integer sequence (char2idx), we need to interpret it as such since it is
    # stored in a tensor that hold objects in order to manage sequences of different lengths in a
    # single tensor.
    sentence = tf.decode_raw(sentence, tf.int32)

    # Apply load_audio to each wav_path of the tensorflow iterator.
    mel_spec, lin_spec = tf.py_func(load_audio, [wav_path], [tf.float32, tf.float32])

    # The shape of the returned values from py_func seems to get lost for some reason.
    mel_spec.set_shape((None, hparams.n_mels))
    lin_spec.set_shape((None, (1 + hparams.n_fft // 2)))

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
            capacity=hparams.train.n_pre_calc_batches,
            bucket_capacities=[hparams.train.n_samples_per_bucket] * (len(buckets) + 1),
            dynamic_pad=True,
            allow_smaller_final_batch=hparams.train.allow_smaller_batches)

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
        # Create a optimizer.
        optimizer = tf.train.AdamOptimizer()

        # Apply gradient clipping by global norm.
        gradients, variables = zip(*optimizer.compute_gradients(loss_op))
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, hparams.train.gradient_clip_norm)

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
    checkpoint_dir = hparams.train.checkpoint_dir

    saver_hook = tf.train.CheckpointSaverHook(
        checkpoint_dir=checkpoint_dir,
        save_secs=hparams.train.checkpoint_save_secs,
    )

    summary_hook = tf.train.SummarySaverHook(
        output_dir=checkpoint_dir,
        save_steps=hparams.train.summary_save_steps,
        summary_op=summary_op
    )

    nan_hook = tf.train.NanTensorHook(
        loss_tensor=loss_op,
        fail_on_nan_loss=True
    )

    counter_hook = tf.train.StepCounterHook(
        output_dir=checkpoint_dir,
        every_n_steps=hparams.train.performance_log_steps
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
        config=session_config,
        checkpoint_dir=checkpoint_dir)

    tf.train.start_queue_runners(sess=session)

    return session


if __name__ == '__main__':
    # Create a dataset loader.
    train_dataset = hparams.train.dataset_loader(dataset_folder=hparams.train.dataset_folder,
                                                 char_dict=hparams.vocabulary_dict,
                                                 fill_dict=False)

    # Create batched placeholders from the dataset.
    placeholders, n_samples = batched_placeholders(dataset=train_dataset,
                                                   max_samples=hparams.train.max_samples,
                                                   n_epochs=hparams.train.n_epochs,
                                                   batch_size=hparams.train.batch_size)

    # Create the Tacotron model.
    tacotron_model = Tacotron(hparams=hparams, inputs=placeholders, training=True)

    # Train the model.
    train(tacotron_model)
