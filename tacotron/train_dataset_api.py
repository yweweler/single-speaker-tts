import math
import time

import numpy as np
import tensorflow as tf

from audio.conversion import ms_to_samples, magnitude_to_decibel, normalize_decibel
from audio.features import mel_scale_spectrogram, linear_scale_spectrogram
from audio.io import load_wav
from tacotron.hparams import hparams
from tacotron.model import Tacotron

tf.logging.set_verbosity(tf.logging.INFO)


def load_entry(entry):
    base_path = '/home/yves-noel/documents/master/projects/datasets/timit/TIMIT/'
    win_len = ms_to_samples(hparams.win_len, hparams.sampling_rate)
    hop_len = ms_to_samples(hparams.win_hop, hparams.sampling_rate)

    out_batch_linear = list()
    out_batch_mel = list()
    for path in entry.tolist():
        wav, sr = load_wav(base_path + path.decode())

        linear_spec = linear_scale_spectrogram(wav, hparams.n_fft, hop_len, win_len).T

        mel_spec = mel_scale_spectrogram(wav, hparams.n_fft, sr, hparams.n_mels,
                                         hparams.mel_fmin, hparams.mel_fmax, hop_len, win_len, 1).T

        dev = 1e-4 / 2
        mel_spec_noisy = mel_spec + np.random.uniform(low=0.0, high=dev, size=np.prod(mel_spec.shape)).reshape(
            mel_spec.shape)
        mel_spec = mel_spec_noisy

        # Convert the linear spectrogram into decibel representation.
        linear_mag = np.abs(linear_spec)
        linear_mag_db = magnitude_to_decibel(linear_mag)
        linear_mag_db = normalize_decibel(linear_mag_db, 20, 100)

        # Convert the mel spectrogram into decibel representation.
        mel_mag = np.abs(mel_spec)
        mel_mag_db = magnitude_to_decibel(mel_mag)
        mel_mag_db = normalize_decibel(mel_mag_db, -7.7, 95.8)

        out_batch_linear.append(linear_mag_db)
        out_batch_mel.append(mel_mag_db)

    return np.array(['text'] * entry.shape[0], dtype=np.object), \
           np.array(out_batch_mel).astype(np.float32), \
           np.array(out_batch_linear).astype(np.float32)


def feedable_train_data(file_list_path, batch_size):
    # For more information on the input pipeline see:
    # https://www.tensorflow.org/versions/master/performance/datasets_performance

    # Read all lines from the file listing.
    with open(file_list_path, 'r') as listing:
        lines = listing.readlines()

    paths = [line.split(',')[0] for line in lines]

    print('Dataset contains {} entries.'.format(len(paths)))
    dataset = tf.data.Dataset.from_tensor_slices(paths)

    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.map(
        lambda filenames: tf.py_func(load_entry, [filenames], [tf.string, tf.float32, tf.float32]),
        num_parallel_calls=4
    )
    dataset.prefetch(64)

    dataset = dataset.repeat(1)
    iterator = dataset.make_one_shot_iterator()

    n_batches = math.ceil(len(paths) / batch_size)

    return iterator, n_batches


def train(checkpoint_dir):
    file_listing_path = '/tmp/train_all.txt'

    n_epochs = 1
    batch_size = 1

    # Checkpoint every 10 minutes.
    checkpoint_save_secs = 60 * 10

    # Save summary every 10 steps.
    summary_save_steps = 10

    model = Tacotron(hparams=hparams)
    inp_mel, inp_linear = model.get_inputs_placeholders()

    loss_op = model.get_loss_op()
    summary_op = model.summary()

    # NOTE: The global step has to be created before the optimizer is created.
    tf.train.create_global_step()

    learning_rate = tf.train.exponential_decay(learning_rate=0.01,
                                               global_step=tf.train.get_global_step(),
                                               decay_steps=100,
                                               decay_rate=0.98)

    # optimizer = tf.train.RMSPropOptimizer(learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Tell the optimizer to minimize the loss function.
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

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

    session_config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
        )
    )

    dataset_start = time.time()

    dataset_iter, n_batches = feedable_train_data(file_listing_path, batch_size)
    features = dataset_iter.get_next()

    dataset_duration = time.time() - dataset_start
    print('Dataset generation: {}s'.format(dataset_duration))

    session = tf.train.SingularMonitoredSession(hooks=[saver_hook, summary_hook, nan_hook],
                                                scaffold=session_scaffold,
                                                config=session_config,
                                                checkpoint_dir=checkpoint_dir)

    train_start = time.time()
    for epoch in range(n_epochs):
        while True:
            try:
                _, mel_batch, linear_batch = session.run(features, feed_dict={
                    inp_mel: np.zeros(shape=(1, 1, hparams.n_mels)),
                    inp_linear: np.zeros(shape=(1, 1, 1 + hparams.n_fft // 2)),
                })

                _, loss_value = session.run([train_op, loss_op], feed_dict={
                    inp_mel: mel_batch,
                    inp_linear: linear_batch
                })
                print(loss_value)

            except tf.errors.OutOfRangeError:
                print('All batches read.')
                break

    train_duration = time.time() - train_start
    print('Training duration: {}s'.format(train_duration))

    session.close()


if __name__ == '__main__':
    train(checkpoint_dir='/tmp/tacotron')
