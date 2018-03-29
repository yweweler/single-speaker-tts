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

    wav, sr = load_wav(base_path + entry.decode())

    linear_spec = linear_scale_spectrogram(wav, hparams.n_fft, hop_len, win_len).T

    mel_spec = mel_scale_spectrogram(wav, hparams.n_fft, sr, hparams.n_mels,
                                     hparams.mel_fmin, hparams.mel_fmax, hop_len, win_len, 1).T

    # dev = 1e-4 / 2
    # mel_spec_noisy = mel_spec + np.random.uniform(low=0.0,
    #                                               high=dev,
    #                                               size=np.prod(mel_spec.shape)).reshape(mel_spec.shape)
    # mel_spec = mel_spec_noisy

    # Convert the linear spectrogram into decibel representation.
    linear_mag = np.abs(linear_spec)
    linear_mag_db = magnitude_to_decibel(linear_mag)
    linear_mag_db = normalize_decibel(linear_mag_db, 20, 100)

    # Convert the mel spectrogram into decibel representation.
    mel_mag = np.abs(mel_spec)
    mel_mag_db = magnitude_to_decibel(mel_mag)
    mel_mag_db = normalize_decibel(mel_mag_db, -7.7, 95.8)

    # ==============================================================================================
    # Tacotron reduction factor.
    # ==============================================================================================
    n_frames = mel_mag_db.shape[0]

    # Calculate how much padding frames have to be added to be a multiple of `reduction`.
    n_padding_frames = hparams.reduction - (n_frames % hparams.reduction) if (
                                                                                     n_frames % hparams.reduction) != 0 else 0

    # Add padding frames to the mel spectrogram.
    mel_mag_db = np.pad(mel_mag_db, [[0, n_padding_frames], [0, 0]], mode="constant")
    mel_mag_db = mel_mag_db.reshape((-1, mel_mag_db.shape[1] * hparams.reduction))

    # Since the magnitude spectrogram has to have the same number of frames we need to add padding.
    linear_mag_db = np.pad(linear_mag_db, [[0, n_padding_frames], [0, 0]], mode="constant")
    linear_mag_db = linear_mag_db.reshape((-1, linear_mag_db.shape[1] * hparams.reduction))
    # ==============================================================================================

    # print("load_entry.mel.shape", np.array(mel_mag_db).astype(np.float32).shape)
    # print("load_entry.linear.shape", np.array(linear_mag_db).astype(np.float32).shape)

    return np.array(mel_mag_db).astype(np.float32), \
           np.array(linear_mag_db).astype(np.float32)


def load_text(text_paths):
    lines = list()
    line_lens = list()

    char_dict = {
        'pad': 0,   # padding
        'eos': 1,   # end of sequence
    }

    base_folder = '/home/yves-noel/documents/master/projects/datasets/timit/TIMIT/'
    for text_path in text_paths:
        with open(base_folder + text_path, 'r') as text_file:
            line = text_file.readline()
            line = line.replace('\n', '')
            line = line.split(' ', 2)[-1]

            # Update character dictionary with the chars contained in the line.
            for char in line:
                if char not in char_dict.keys():
                    # Character is not contained in the dictionary, we need to add it.
                    char_dict[char] = len(char_dict.keys())

            # Convert each character into its dictionary index and add the eos token to the end.
            idx = [char_dict[char] for char in line]
            idx.append(char_dict['eos'])

            # Append str. representation so that tf.Tensor handles this as an collection of objects.
            # This allows us to store sequences of different length in a single tensor.
            lines.append(np.array(idx, dtype=np.uint16).tostring())
            line_lens.append(len(line))

    print('char_dict:', char_dict)
    return lines, line_lens


def train_data_buckets(file_list_path, n_epochs, batch_size):
    n_threads = 4

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
    print('len(wav_paths)', len(wav_paths))

    # Convert everything into tf.Tensor objects for queue based processing.
    wav_paths = tf.convert_to_tensor(wav_paths)
    sentence_lengths = tf.convert_to_tensor(sentence_lengths)
    sentences = tf.convert_to_tensor(sentences)

    # Create a queue based iterator that yields tuples to process.
    wav_path, sentence_length, sentence = tf.train.slice_input_producer(
        [wav_paths, sentence_lengths, sentences],
        capacity=n_threads * batch_size,
        num_epochs=n_epochs,
        shuffle=False)

    # The sentence is a integer sequence (char2idx), we need to interpret it as such since it is stored in
    # a tensor that hold objects in order to manage sequences of different lengths in a single tensor.
    sentence = tf.decode_raw(sentence, tf.uint16)

    # Apply load_entry to each wav_path of the tensorflow iterator.
    mel, mag = tf.py_func(load_entry, [wav_path], [tf.float32, tf.float32])

    # The shape of the returned values from py_func seems to get lost for some reason.
    mel.set_shape((None, hparams.n_mels * hparams.reduction))
    mag.set_shape((None, (1 + hparams.n_fft // 2) * hparams.reduction))

    # TODO: tf.train.batch also supports dynamic per batch padding using 'dynamic_pad=True'
    # mels, mags = tf.train.batch([mel, mag], batch_size=batch_size, capacity=64, num_threads=4)

    print('n_buckets: {} + 2'.format(len([i for i in range(minlen + 1, maxlen + 1, 4)])))
    batch_sequence_lengths, (sents, mels, mags) = tf.contrib.training.bucket_by_sequence_length(
        input_length=sentence_length,
        tensors=[sentence, mel, mag],
        batch_size=batch_size,
        bucket_boundaries=[i for i in range(minlen + 1, maxlen + 1, 4)],
        num_threads=n_threads,
        capacity=n_threads * batch_size,
        dynamic_pad=True,
        allow_smaller_final_batch=True)

    n_batches = int(math.ceil(len(lines) / batch_size))

    print('batched.sentence.shape', sents.shape)
    print('batched.mel.shape', mels.shape)
    print('batched.mag.shape', mags.shape)

    return batch_sequence_lengths, sents, mels, mags, n_batches


def train(checkpoint_dir):
    file_listing_path = 'data/train_all.txt'

    n_epochs = 10
    batch_size = 4

    # Checkpoint every 10 minutes.
    checkpoint_save_secs = 60 * 10

    # Save summary every 10 steps.
    summary_save_steps = 10
    summary_counter_steps = 100

    dataset_start = time.time()
    lengths_iter, sent_iter, mel_iter, linear_iter, n_batches = train_data_buckets(
        file_listing_path,
        n_epochs,
        batch_size)
    dataset_duration = time.time() - dataset_start
    print('Dataset generation: {}s'.format(dataset_duration))

    # For debugging purposes only.
    mel_iter = tf.Print(mel_iter, [sent_iter], summarize=30)

    model = Tacotron(hparams=hparams, inputs=(sent_iter, mel_iter, linear_iter, lengths_iter))

    loss_op = model.get_loss_op()

    # NOTE: The global step has to be created before the optimizer is created.
    tf.train.create_global_step()

    optimizer = tf.train.AdamOptimizer()

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

    train_start = time.time()

    while not session.should_stop():
        try:
            session.run([train_op])
        except tf.errors.OutOfRangeError:
            print('All batches read.')
            break

    train_duration = time.time() - train_start
    print('Training duration: {}s'.format(train_duration))

    session.close()


if __name__ == '__main__':
    train(checkpoint_dir='/tmp/tacotron')
