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


def apply_reduction(mel_mag_db, linear_mag_db, reduction_factor):
    """
    Reduces `reduction_factor` consecutive frames of the mel and linear spectrogram's into single
    frames by means of concatenating them. Automatically pads with zero frames in the time
    dimension such that the number of frames is a multiple of `reduction_factor`.

    Arguments:
        mel_mag_db (np.ndarray):
            Mel scale spectrogram to be reduced. The shape is expected to be shape=(T, n_mels),
            with T being the number of frames.

        linear_mag_db (np.ndarray):
            Linear scale spectrogram to be reduced. The shape is expected to be
            shape=(T, 1 + n_fft // 2), with T being the number of frames.

        reduction_factor (int):
            The number of consecutive frames that should be reduced to a single frame.

    Returns:
        (mel_mag_db, linear_mag_db):
            mel_mag_db (np.ndarray):
                Reduced Mel scale spectrogram. The shape is shape=(T // r, n_mels * r),
                with T being the number of frames.
            linear_mag_db (np.ndarray):
                Reduced linear scale spectrogram. The shape is shape=(T // r, (1 + n_fft // 2) * r),
                with T being the number of frames.
    """
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
    win_len = ms_to_samples(hparams.win_len, hparams.sampling_rate)
    hop_len = ms_to_samples(hparams.win_hop, hparams.sampling_rate)

    wav, sr = load_wav(file_path.decode())

    # TODO: resample from 22050 Hz to 16000 Hz?

    # TODO: Determine a better silence reference level for the lj dataset.
    # Remove silence at the beginning and end of the wav so the network does not have to learn
    # some random initial silence delay after which it is allowed to speak.
    # wav, _ = trim_silence(wav)

    linear_spec = linear_scale_spectrogram(wav, hparams.n_fft, hop_len, win_len).T

    mel_spec = mel_scale_spectrogram(wav, hparams.n_fft, sr, hparams.n_mels,
                                     hparams.mel_fmin, hparams.mel_fmax, hop_len, win_len, 1).T

    # Convert the linear spectrogram into decibel representation.
    linear_mag = np.abs(linear_spec)
    linear_mag_db = magnitude_to_decibel(linear_mag)
    linear_mag_db = normalize_decibel(linear_mag_db, 35.7, 100)  # TODO: Refactor numbers.
    # => linear_mag_db.shape = (n_frames, 1 + n_fft // 2)

    # Convert the mel spectrogram into decibel representation.
    mel_mag = np.abs(mel_spec)
    mel_mag_db = magnitude_to_decibel(mel_mag)
    mel_mag_db = normalize_decibel(mel_mag_db, 6.0, 100)  # TODO: Refactor numbers.
    # => mel_mag_db.shape = (n_frames, n_mels)

    # print("[ORIGINAL] load_audio.mel_spec.shape", np.array(mel_mag_db).astype(np.float32).shape)

    # Tacotron reduction factor.
    if hparams.reduction > 1:
        # mel_mag_db.shape => (T // r, n_mels * r)
        # linear_mag_db. shape => (T // r, (1 + n_fft // 2) * r)
        mel_mag_db, linear_mag_db = apply_reduction(mel_mag_db, linear_mag_db, hparams.reduction)

    # print("[REDUCED] load_audio.mel_spec.shape", np.array(mel_mag_db).astype(np.float32).shape)

    return np.array(mel_mag_db).astype(np.float32), \
           np.array(linear_mag_db).astype(np.float32)


def batched_placeholders(dataset, n_epochs, batch_size):
    n_threads = hparams.train.n_threads

    sentences, sentence_lengths, wav_paths = dataset.load(max_samples=hparams.train.max_samples)
    max_len, min_len = max(sentence_lengths), min(sentence_lengths)

    # Convert everything into tf.Tensor objects for queue based processing.
    sentences = tf.convert_to_tensor(sentences)
    sentence_lengths = tf.convert_to_tensor(sentence_lengths)
    wav_paths = tf.convert_to_tensor(wav_paths)

    # Create a queue based iterator that yields tuples to process.
    sentence, sentence_length, wav_path = tf.train.slice_input_producer(
        [sentences, sentence_lengths, wav_paths],
        capacity=n_threads * batch_size,
        num_epochs=n_epochs,
        shuffle=False)

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
            capacity=n_threads * 2,  # Number of batches to prepare.
            bucket_capacities=[batch_size * 4] * (len(buckets) + 1),  # Samples per bucket.
            dynamic_pad=True,
            allow_smaller_final_batch=True)

    print('batched.ph_sentence_length', ph_sentence_length.shape, ph_sentence_length)
    print('batched.ph_sentences.shape', ph_sentences.shape, ph_sentences)
    print('batched.ph_mel_specs.shape', ph_mel_specs.shape, ph_mel_specs)
    print('batched.ph_lin_specs.shape', ph_lin_specs.shape, ph_lin_specs)
    print('batched.ph_time_frames', ph_time_frames.shape, ph_time_frames)

    # get the total number of samples in the dataset.
    n_samples = sentence_lengths

    return (ph_sentences, ph_sentence_length, ph_mel_specs, ph_lin_specs, ph_time_frames), n_samples


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

    # TODO: Not sure what the exact benefit of a scaffold is since it does not hold very much data.
    session_scaffold = tf.train.Scaffold(
        init_op=tf.global_variables_initializer(),
        summary_op=summary_op
    )

    saver_hook = tf.train.CheckpointSaverHook(
        checkpoint_dir=checkpoint_dir,
        save_secs=hparams.train.checkpoint_save_secs,
        scaffold=session_scaffold
    )

    summary_hook = tf.train.SummarySaverHook(
        output_dir=checkpoint_dir,
        save_steps=hparams.train.summary_save_steps,
        scaffold=session_scaffold
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
        scaffold=session_scaffold,
        config=session_config,
        checkpoint_dir=checkpoint_dir)

    tf.train.start_queue_runners(sess=session)

    return session


if __name__ == '__main__':
    # Create a dataset loader.
    dataset = hparams.train.dataset_loader(dataset_folder=hparams.train.dataset_folder,
                                           char_dict=hparams.vocabulary_dict,
                                           fill_dict=False)

    # Create batched placeholders from the dataset.
    placeholders, n_samples = batched_placeholders(dataset=dataset,
                                                   n_epochs=hparams.train.n_epochs,
                                                   batch_size=hparams.train.batch_size)

    # Create the Tacotron model.
    tacotron_model = Tacotron(hparams=hparams, inputs=placeholders, training=True)

    # Train the model.
    train(tacotron_model)
