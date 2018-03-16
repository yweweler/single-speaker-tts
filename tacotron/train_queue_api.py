import math

import numpy as np
import tensorflow as tf

from audio.conversion import ms_to_samples, magnitude_to_decibel
from audio.features import mel_scale_spectrogram, linear_scale_spectrogram
from audio.io import load_wav
from tacotron.hparams import hparams
from tacotron.model import Tacotron


def load_entry(entry):
    base_path = '/home/yves-noel/documents/master/projects/datasets/timit/TIMIT/'
    # print('-' * 64)
    # print('load_batch: {}, type: {}'.format(entry.shape, entry.dtype))
    # print(entry)
    # print('-' * 64)

    win_len = ms_to_samples(25.0, 16000)
    hop_len = ms_to_samples(8.0, 16000)

    out_batch_linear = list()
    out_batch_mel = list()
    for path in entry.tolist():
        wav, sr = load_wav(base_path + path.decode())
        # wav, sr = load_wav(base_path + path)

        linear_spec = linear_scale_spectrogram(wav, 512, hop_len, win_len).T
        mel_spec = mel_scale_spectrogram(wav, 512, sr, 80, 0, 8000, hop_len, win_len, 1).T

        # Convert the linear spectrogram into decibel representation.
        linear_mag = np.abs(linear_spec)
        linear_mag_db = magnitude_to_decibel(linear_mag)
        # linear_mag_db = normalize_decibel(linear_mag_db, 20, 100)
        # print('linear_spec', linear_mag_db.shape)

        # t = linear_mag_db.shape[0]
        # r = 1
        # num_paddings = r - (t % r) if t % r != 0 else 0  # for reduction
        # linear_mag_db = np.pad(linear_mag_db, [[0, num_paddings], [0, 0]], mode="constant")

        # Convert the mel spectrogram into decibel representation.
        mel_mag = np.abs(mel_spec)
        mel_mag_db = magnitude_to_decibel(mel_mag)
        # mel_mag_db = normalize_decibel(mel_mag_db, -7.7, 95.8)
        # print('mel_spec', mel_mag_db.shape)

        out_batch_linear.append(linear_mag_db)
        out_batch_mel.append(mel_mag_db)

    # np_wav_paths = entry[:, 0]
    # np_txt_paths = entry[:, 1]
    # # Load the text strings.
    # txt_data = list()
    # for path in np_txt_paths.tolist():
    #     with open(base_path + path.decode(), 'r') as txt_file:
    #         raw_line = txt_file.readline()
    #         raw_line = raw_line.replace('\n', '')
    #         txt = raw_line.split(' ', 2)[-1]
    #         txt_data.append(txt)
    #
    # # return np.array(txt_data, dtype=np.object)

    linear_column = np.array(out_batch_mel)
    # mel_column = np.array(out_batch_linear).astype(np.float32)
    # print("linear_column.shape", linear_column.shape)
    # print("mel_column.shape", mel_column.shape)
    # return mel_column, linear_column
    return np.array(['text'] * entry.shape[0], dtype=np.object), \
           np.array(out_batch_linear).astype(np.float32), \
           np.array(out_batch_mel).astype(np.float32)
    # np.array(out_batch_linear),\
    # np.array(out_batch_mel)


def feedable_train_data(file_list_path, batch_size):
    # Read all lines from the file listing.
    with open(file_list_path, 'r') as listing:
        lines = listing.readlines()

    paths = [line.split(',')[0] for line in lines]

    print('Dataset contains {} entries.'.format(len(paths)))
    dataset = tf.data.Dataset.from_tensor_slices(paths)

    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.map(
        lambda filenames: tf.py_func(load_entry, [filenames], [tf.string, tf.float32, tf.float32])
    )

    dataset = dataset.repeat(1)
    iterator = dataset.make_one_shot_iterator()

    n_batches = math.ceil(len(paths) / batch_size)

    return iterator, n_batches


def train_data(file_list_path, batch_size):
    # TODO: Force the code to run on the cpu not matter what device is selected for the actual training process.

    # Read all lines from the file listing.
    with open(file_list_path, 'r') as listing:
        lines = listing.readlines()

    # wav_paths = [line.split(',')[0] for line in lines]
    # txt_paths = [line.split(',')[1] for line in lines]
    paths = [line.split(',')[0] for line in lines]

    print('Dataset contains {} entries.'.format(len(paths)))
    dataset = tf.data.Dataset.from_tensor_slices(paths)

    # Shuffle the entire dataset at the beginning of each epoch.
    # dataset = dataset.shuffle(buffer_size=len(paths))

    # Map and batch the dataset in parallel. This is more efficient than doing this manually using two steps.
    # dataset = dataset.apply(
    #     tf.contrib.data.map_and_batch(map_func=load_entry, batch_size=batch_size, num_parallel_batches=2)
    # )

    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.map(
        lambda filenames: tf.py_func(load_entry, [filenames], [tf.string, tf.float32, tf.float32])
    )

    # Cache database entries in memory.
    # dataset = dataset.cache()

    # Prefetch an entire batch so the training process does not come to an halt.
    # dataset = dataset.prefetch(buffer_size=batch_size)

    # Stop providing batches after one epoch.
    dataset = dataset.repeat(1)

    iterator = dataset.make_initializable_iterator()

    # Calculate the number of batches in one epoch (used to monitor the progress).
    n_batches = math.ceil(len(paths) / batch_size)

    return iterator, n_batches


def load_text(text_paths):
    lines = list()
    line_lens = list()
    base_folder = '/home/yves-noel/documents/master/projects/datasets/timit/TIMIT/'
    for text_path in text_paths:
        with open(base_folder + text_path, 'r') as text_file:
            line = text_file.readline()
            line = line.replace('\n', '')
            line = line.split(' ', 2)[-1]
            lines.append(line)
            line_lens.append(len(line))

    max_len = max(line_lens)
    texts = np.zeros((len(lines), max_len), np.int32)
    # TODO: char2idx for all lines.

    return texts, line_lens


def train_data_bueckets(file_list_path, batch_size):
    # TODO: Stupid shitty queue runners have to be started in some face way at some point I do not know x(.
    # Read all lines from the file listing.
    with open(file_list_path, 'r') as listing:
        lines = listing.readlines()

    # wav_paths = [line.split(',')[0] for line in lines]
    # txt_paths = [line.split(',')[1] for line in lines]
    wav_paths = [line.split(',')[0] for line in lines]

    # _, mel, mag = tf.py_func(load_entry, [wav_paths], [tf.string, tf.float32, tf.float32])
    _, mel, mag = load_entry(np.array(wav_paths, dtype=np.object))

    text_paths = [path.split(',')[1] for path in lines]
    sentences, sentence_lengths = load_text(text_paths)
    maxlen, minlen = max(sentence_lengths), min(sentence_lengths)
    print('minlen', minlen)
    print('maxlen', maxlen)
    print('len(sentence_lengths)', len(sentence_lengths))

    # sentences = tf.convert_to_tensor(sentences)
    # mel = tf.convert_to_tensor(mel, dtype=tf.float32)
    # mag = tf.convert_to_tensor(mag)
    print('sentences.shape', sentences.shape)
    print('mel.shape', mel.shape)
    print('mag.shape', mag.shape)

    _, (sents, mels, mags) = tf.contrib.training.bucket_by_sequence_length(
        input_length=sentence_lengths,
        tensors=[sentences, mel, mag],
        batch_size=batch_size,
        bucket_boundaries=[i for i in range(minlen + 1, maxlen, 1)],
        num_threads=4,
        capacity=batch_size * 4,
        dynamic_pad=True)

    for s, m, l in zip(sents, mels, mags):
        print(s.shape, m.shape, l.shape)


def train(checkpoint_dir):
    file_listing_path = '/tmp/train_all.txt'

    n_epochs = 1
    batch_size = 1

    # Checkpoint every 10 minutes.
    checkpoint_save_secs = 60 * 10

    # Save summary every 10 steps.
    summary_save_steps = 10

    inputs = Tacotron.inputs()
    model = Tacotron(hparams=hparams, inputs=inputs)

    dataset_iter, n_batches = feedable_train_data(file_listing_path, batch_size)
    # print(dataset_iter.get_next())
    _, inp_mel, inp_linear = dataset_iter.get_next()

    loss_op = model.loss()
    summary_op = model.summary()

    # NOTE: The global step has to be created before the optimizer is created.
    tf.train.create_global_step()

    optimizer = tf.train.AdamOptimizer()

    # Tell the optimizer to minimize the loss function.
    train_op = optimizer.minimize(model.loss(), global_step=tf.train.get_global_step())

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
    # TODO: inter_op_parallelism_threads=16

    session = tf.train.SingularMonitoredSession(hooks=[saver_hook, summary_hook, nan_hook],
                                                scaffold=session_scaffold,
                                                config=session_config,
                                                checkpoint_dir=checkpoint_dir)

    # TODO: Is this necessary? I guess the MonitoredSession does this itself using `session_scaffold.init_op`.
    # session.run(tf.global_variables_initializer())

    # TODO: Is this necessary? I guess the MonitoredSession will load the network from a checkpoint itself.
    # model.load(checkpoint_dir)

    # session.run(dataset_iter.initializer)

    # TODO: Feed batches to the network and update gradients.
    for epoch in range(n_epochs):
        print('\n')
        print('=' * 64)
        print('Epoch', epoch + 1)
        print('=' * 64)

        # Reset the dataset iterator to begin an epoch.
        # session.run(dataset_iter.initializer)
        # Create a progress bar to display the training progress.
        # progress_bar = trange(n_batches, dynamic_ncols=True, leave=True)

        while True:
            try:
                # batch = session.run(batch_iter)
                # print('=' * 64)
                # print('batch:\n{}'.format(batch))
                # print('=' * 64)
                # print('')
                session.run(train_op)

            except tf.errors.OutOfRangeError:
                print('')
                print('All batches read.')
                print('=' * 64)
                break

            # progress_bar.set_description('Epoch {}'.format(epoch + 1))
            # batches.set_postfix(loss=0.0)
            # time.sleep(0.15)

    session.close()


if __name__ == '__main__':
    train(checkpoint_dir='/tmp/tacotron')
