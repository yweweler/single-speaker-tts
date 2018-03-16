import time

import numpy as np
import tensorflow as tf

from audio.conversion import ms_to_samples, magnitude_to_decibel, normalize_decibel
from audio.features import mel_scale_spectrogram, linear_scale_spectrogram
from audio.io import load_wav
from tacotron.hparams import hparams
from tacotron.model import Tacotron


def load_entry(entry):
    print('=' * 64)
    print(entry.shape)
    print(entry)

    base_path = '/home/yves-noel/documents/master/projects/datasets/timit/TIMIT/'
    win_len = ms_to_samples(hparams.win_len, hparams.sampling_rate)
    hop_len = ms_to_samples(hparams.win_hop, hparams.sampling_rate)

    wav, sr = load_wav(base_path + entry.decode())

    linear_spec = linear_scale_spectrogram(wav, hparams.n_fft, hop_len, win_len).T

    mel_spec = mel_scale_spectrogram(wav, hparams.n_fft, sr, hparams.n_mels,
                                     hparams.mel_fmin, hparams.mel_fmax, hop_len, win_len, 1).T

    dev = 1e-4 / 2
    mel_spec_noisy = mel_spec + np.random.uniform(low=0.0,
                                                  high=dev,
                                                  size=np.prod(mel_spec.shape)).reshape(mel_spec.shape)
    mel_spec = mel_spec_noisy

    # Convert the linear spectrogram into decibel representation.
    linear_mag = np.abs(linear_spec)
    linear_mag_db = magnitude_to_decibel(linear_mag)
    linear_mag_db = normalize_decibel(linear_mag_db, 20, 100)

    # Convert the mel spectrogram into decibel representation.
    mel_mag = np.abs(mel_spec)
    mel_mag_db = magnitude_to_decibel(mel_mag)
    mel_mag_db = normalize_decibel(mel_mag_db, -7.7, 95.8)

    return np.array(linear_mag_db).astype(np.float32),\
           np.array(mel_mag_db).astype(np.float32)


def load_text(text_paths):
    lines = list()
    line_lens = list()

    base_folder = '/home/yves-noel/documents/master/projects/datasets/timit/TIMIT/'
    for text_path in text_paths:
        with open(base_folder + text_path, 'r') as text_file:
            line = text_file.readline()
            line = line.replace('\n', '')
            line = line.split(' ', 2)[-1]
            # Append string representation so that tf.Tensor handles this as an collection of objects.
            # This allows us to store sequences of different length in a single tensor.
            # TODO: char2idx for all lines.
            lines.append(np.zeros(shape=(len(line)), dtype=np.int32).tostring())
            line_lens.append(len(line))

    return lines, line_lens


def train_data_bueckets(file_list_path, batch_size):
    # Read all lines from the file listing.
    with open(file_list_path, 'r') as listing:
        lines = listing.readlines()

    # Get wav and text paths from the listing lines.
    wav_paths = [line.split(',')[0] for line in lines]
    text_paths = [path.split(',')[1] for path in lines]

    # Load the sentences from files and determine the sentence lengths.
    sentences, sentence_lengths = load_text(text_paths)
    maxlen, minlen = max(sentence_lengths), min(sentence_lengths)

    print('minlen', minlen)
    print('maxlen', maxlen)
    print('len(sentence_lengths)', len(sentence_lengths))

    # Convert everything into tf.Tensor objects for queue based processing.
    wav_paths = tf.convert_to_tensor(wav_paths)
    sentence_lengths = tf.convert_to_tensor(sentence_lengths)
    sentences = tf.convert_to_tensor(sentences)

    # Create a queue based iterator that yields tuples to process.
    wav_path, sentence_length, sentence = tf.train.slice_input_producer([wav_paths, sentence_lengths, sentences],
                                                                        shuffle=False)

    # The sentence is a integer sequence (char2idx), we need to interpret it as such since it is stored in
    # a tensor that hold objects in order to manage sequences of different lengths in a single tensor.
    sentence = tf.decode_raw(sentence, tf.int32)

    # Apply load_entry to each wav_path of the tensorflow iterator.
    mel, mag = tf.py_func(load_entry, [wav_path], [tf.float32, tf.float32])

    # TODO: The shape of the returned values from py_func seems to get lost for some reason.

    print('sentences.shape', sentences.shape)
    print('mel.shape', mel.shape)
    print('mag.shape', mag.shape)

    # _, (sents, mels, mags) = tf.contrib.training.bucket_by_sequence_length(
    #     input_length=sentence_lengths,
    #     tensors=[sentence, mel, mag],
    #     batch_size=batch_size,
    #     bucket_boundaries=[i for i in range(minlen + 1, maxlen, 1)],
    #     num_threads=4,
    #     capacity=batch_size * 4,
    #     dynamic_pad=True)

    return sentence, mel, mag # sents, mels, mags


def train(checkpoint_dir):
    file_listing_path = '/tmp/train_all.txt'

    n_epochs = 1
    batch_size = 1

    # Checkpoint every 10 minutes.
    checkpoint_save_secs = 60 * 10

    # Save summary every 10 steps.
    summary_save_steps = 10

    dataset_start = time.time()
    sent_iter, mel_iter, linear_iter = train_data_bueckets(file_listing_path, batch_size)
    dataset_duration = time.time() - dataset_start
    print('Dataset generation: {}s'.format(dataset_duration))

    model = Tacotron(hparams=hparams, inputs=(mel_iter, linear_iter))

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

    session = tf.train.SingularMonitoredSession(hooks=[saver_hook, summary_hook, nan_hook],
                                                scaffold=session_scaffold,
                                                config=session_config,
                                                checkpoint_dir=checkpoint_dir)

    train_start = time.time()

    # Start the data queue
    tf.train.start_queue_runners()

    for epoch in range(n_epochs):
        while True:
            try:
                _, loss_value = session.run([train_op, loss_op])
                print(loss_value)

            except tf.errors.OutOfRangeError:
                print('All batches read.')
                break

    train_duration = time.time() - train_start
    print('Training duration: {}s'.format(train_duration))

    session.close()


if __name__ == '__main__':
    train(checkpoint_dir='/tmp/tacotron')
