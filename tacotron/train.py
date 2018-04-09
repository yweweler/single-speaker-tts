import time

import numpy as np
import tensorflow as tf

from audio.conversion import ms_to_samples, magnitude_to_decibel, normalize_decibel
from audio.effects import trim_silence
from audio.features import mel_scale_spectrogram, linear_scale_spectrogram
from audio.io import load_wav
from datasets.lj_speech import LJSpeechDatasetHelper
from tacotron.hparams import hparams
from tacotron.model import Tacotron

# Hack to force tensorflow to run on the CPU. (see issue #152)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

tf.logging.set_verbosity(tf.logging.INFO)


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
    linear_mag_db = normalize_decibel(linear_mag_db, 35.7, 100)

    # Convert the mel spectrogram into decibel representation.
    mel_mag = np.abs(mel_spec)
    mel_mag_db = magnitude_to_decibel(mel_mag)
    mel_mag_db = normalize_decibel(mel_mag_db, 6.0, 100)

    # ==============================================================================================
    # Tacotron reduction factor.
    # ==============================================================================================
    # n_frames = mel_mag_db.shape[0]
    #
    # # Calculate how much padding frames have to be added to be a multiple of `reduction`.
    # n_padding_frames = hparams.reduction - (n_frames % hparams.reduction) if (
    #   n_frames % hparams.reduction) != 0 else 0
    #
    # # Add padding frames to the mel spectrogram.
    # mel_mag_db = np.pad(mel_mag_db, [[0, n_padding_frames], [0, 0]], mode="constant")
    # mel_mag_db = mel_mag_db.reshape((-1, mel_mag_db.shape[1] * hparams.reduction))
    #
    # # Since the magnitude spectrogram has to have the same num. of frames we need to add padding.
    # linear_mag_db = np.pad(linear_mag_db, [[0, n_padding_frames], [0, 0]], mode="constant")
    # linear_mag_db = linear_mag_db.reshape((-1, linear_mag_db.shape[1] * hparams.reduction))
    # ==============================================================================================

    # print("load_audio.mel_spec.shape", np.array(mel_mag_db).astype(np.float32).shape)
    # print("load_audio.lin_spec.shape", np.array(linear_mag_db).astype(np.float32).shape)

    # print('loaded:', file_path)

    return np.array(mel_mag_db).astype(np.float32), \
           np.array(linear_mag_db).astype(np.float32)


def batched_placeholders(dataset, n_epochs, batch_size):
    n_threads = 4

    sentences, sentence_lengths, wav_paths = dataset.load(max_samples=25)
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
    mel_spec.set_shape((None, hparams.n_mels * hparams.reduction))
    lin_spec.set_shape((None, (1 + hparams.n_fft // 2) * hparams.reduction))

    # Get the number spectrogram time-steps (used as the number of time frames when generating).
    n_time_frames = tf.shape(mel_spec)[0]

    buckets = [i for i in range(min_len + 1, max_len + 1, 8)]
    print('n_buckets: {} + 2'.format(len(buckets)))

    # Just batch data no matter what sequence length.
    # ph_sentence_length, ph_sentences, ph_mel_specs, ph_lin_specs, ph_time_frames = tf.train.batch(
    #     tensors=[
    #         sentence_length,
    #         sentence,
    #         mel_spec,
    #         lin_spec,
    #         n_time_frames
    #     ],
    #     batch_size=batch_size,
    #     capacity=20,
    #     num_threads=n_threads,
    #     dynamic_pad=True
    # )

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


def train(checkpoint_dir):
    init_char_dict = {
        'pad': 0,  # padding
        'eos': 1,  # end of sequence
        'p': 2,
        'r': 3,
        'i': 4,
        'n': 5,
        't': 6,
        'g': 7,
        ' ': 8,
        'h': 9,
        'e': 10,
        'o': 11,
        'l': 12,
        'y': 13,
        's': 14,
        'w': 15,
        'c': 16,
        'a': 17,
        'd': 18,
        'f': 19,
        'm': 20,
        'x': 21,
        'b': 22,
        'v': 23,
        'u': 24,
        'k': 25,
        'j': 26,
        'z': 27,
        'q': 28,
    }

    dataset = LJSpeechDatasetHelper(dataset_folder='/home/yves-noel/downloads/LJSpeech-1.1',
                                    char_dict=init_char_dict,
                                    fill_dict=False)

    n_epochs = 2000
    batch_size = 4

    # Checkpoint every 10 minutes.
    checkpoint_save_secs = 60 * 10

    # Save summary every 100 steps.
    summary_save_steps = 5
    summary_counter_steps = 5

    dataset_start = time.time()
    placeholders, n_samples = batched_placeholders(dataset, n_epochs, batch_size)
    dataset_duration = time.time() - dataset_start

    print('Dataset generation: {}s'.format(dataset_duration))

    model = Tacotron(hparams=hparams,
                     inputs=placeholders,
                     training=True)

    loss_op = model.get_loss_op()

    # NOTE: The global step has to be created before the optimizer is created.
    tf.train.create_global_step()

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer()

        # Apply gradient clipping and collect gradient summaries.
        # gradients, variables = zip(*optimizer.compute_gradients(loss_op))
        # clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        #
        # for grad_tensor in gradients:
        #     tf.summary.histogram('gradients', grad_tensor)
        #
        # # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
        # # https://github.com/tensorflow/tensorflow/issues/1122
        # with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #     optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
        #                                          global_step=tf.train.get_global_step())

        # Tell the optimizer to minimize the loss function.
        train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

    summary_op = model.summary()

    # TODO: Not sure what the exact benefit of a scaffold is since it does not hold very much data.
    session_scaffold = tf.train.Scaffold(
        init_op=tf.global_variables_initializer(),
        summary_op=summary_op
    )

    saver_hook = tf.train.CheckpointSaverHook(
        checkpoint_dir=checkpoint_dir,
        save_secs=checkpoint_save_secs,
        scaffold=session_scaffold
    )

    summary_hook = tf.train.SummarySaverHook(
        output_dir=checkpoint_dir,
        save_steps=summary_save_steps,
        scaffold=session_scaffold
    )

    nan_hook = tf.train.NanTensorHook(
        loss_tensor=loss_op,
        fail_on_nan_loss=True
    )

    counter_hook = tf.train.StepCounterHook(
        output_dir=checkpoint_dir,
        every_n_steps=summary_counter_steps
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

    train_start = time.time()

    # _global_step = tf.train.get_global_step()
    while not session.should_stop():
        try:
            # session.run([_global_step, loss_op, optimize])
            session.run([train_op])
        except tf.errors.OutOfRangeError:
            break

    train_duration = time.time() - train_start

    print('Training duration: {:.3f}min.'.format(train_duration / 60.0))

    session.close()


if __name__ == '__main__':
    train(checkpoint_dir='/tmp/tacotron/ljspeech_all_samples')
