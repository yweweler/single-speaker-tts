import math
import os

import numpy as np
import tensorflow as tf

from audio.conversion import ms_to_samples, magnitude_to_decibel, normalize_decibel, \
    inv_normalize_decibel, decibel_to_magnitude
from audio.effects import trim_silence
from audio.features import mel_scale_spectrogram, linear_scale_spectrogram
from audio.io import load_wav, save_wav
from audio.synthesis import spectrogram_to_wav
from tacotron.hparams import hparams
from tacotron.model import Tacotron

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

tf.logging.set_verbosity(tf.logging.INFO)


def load_entry(entry):
    base_path = '/home/yves-noel/documents/master/projects/datasets/timit/TIMIT/'
    win_len = ms_to_samples(hparams.win_len, hparams.sampling_rate)
    hop_len = ms_to_samples(hparams.win_hop, hparams.sampling_rate)

    wav, sr = load_wav(base_path + entry.decode())

    # Remove silence at the beginning and end of the wav so the network does not have to learn
    # some random initial silence delay after which it is allowed to speak.
    wav, _ = trim_silence(wav)

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
        'pad': 0,  # padding
        'eos': 1,  # end of sequence
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
            lines.append(np.array(idx, dtype=np.int32).tostring())
            line_lens.append(len(line))

    print('char_dict:', len(char_dict))
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
    sentence = tf.decode_raw(sentence, tf.int32)

    # Apply load_entry to each wav_path of the tensorflow iterator.
    mel, mag = tf.py_func(load_entry, [wav_path], [tf.float32, tf.float32])

    # The shape of the returned values from py_func seems to get lost for some reason.
    mel.set_shape((None, hparams.n_mels * hparams.reduction))
    mag.set_shape((None, (1 + hparams.n_fft // 2) * hparams.reduction))

    # Get the number spectrogram time-steps (later used as sequence lengths for the spectrograms).
    time_steps = tf.shape(mel)[0]

    # TODO: tf.train.batch also supports dynamic per batch padding using 'dynamic_pad=True'
    # mels, mags = tf.train.batch([mel, mag], batch_size=batch_size, capacity=64, num_threads=4)

    print('n_buckets: {} + 2'.format(len([i for i in range(minlen + 1, maxlen + 1, 4)])))
    batch_sequence_lengths, (sents, mels, mags, steps) = \
        tf.contrib.training.bucket_by_sequence_length(
            input_length=sentence_length,
            tensors=[sentence, mel, mag, time_steps],
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
    print('batched.steps', steps)

    return batch_sequence_lengths, sents, mels, mags, steps, n_batches


def model_placeholders(max_len):
    inp_sentences = tf.placeholder(dtype=tf.int32, shape=(1, max_len), name='ph_inp_sentences')
    inp_mel_spec = tf.placeholder(dtype=tf.float32, shape=(1, 1000, 80))
    inp_linear_spec = tf.placeholder(dtype=tf.float32)
    seq_lengths = tf.placeholder(dtype=tf.int32)
    inp_time_steps = tf.placeholder(dtype=tf.int32)

    return inp_sentences, inp_mel_spec, inp_linear_spec, seq_lengths, inp_time_steps


def evaluate(checkpoint_dir):
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    max_len = 60

    placeholders = model_placeholders(max_len)
    model = Tacotron(hparams=hparams, inputs=placeholders, training=False)
    saver = tf.train.Saver()

    def pad_sentence(_sentence, _max_len):
        pad = _max_len - len(_sentence)
        _sentence.extend([0] * pad)
        return _sentence

    # TODO: Load sentences.
    sentences = [
        [40, 4, 24, 4, 14, 26, 14, 16, 17, 5, 16, 9, 5, 6, 16, 13, 18, 4, 11, 27, 5, 15, 3, 4,
         8, 5, 13, 4, 15, 5, 15, 3, 4, 5, 28, 14, 11, 4, 20, 1],
        [21, 9, 16, 22, 15, 5, 6, 13, 12, 5, 23, 4, 5, 15, 9, 5, 24, 6, 11, 11, 8, 5, 6, 16, 5,
         9, 14, 19, 8, 5, 11, 6, 17, 5, 19, 14, 12, 4, 5, 15, 3, 6, 15, 20, 1]
    ]

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        print('Restoring model...')
        saver.restore(session, checkpoint_file)
        print('Restoring finished')

        win_len = ms_to_samples(hparams.win_len, sampling_rate=hparams.sampling_rate)
        win_hop = ms_to_samples(hparams.win_hop, sampling_rate=hparams.sampling_rate)
        n_fft = hparams.n_fft

        with tf.device('/cpu:0'):
            for i, sentence in enumerate(sentences):
                print('Sentence: {} len: {}'.format(i, len(sentence)))
                shizzle = pad_sentence(sentence, max_len)
                print(shizzle)
                test = np.array([shizzle], dtype=np.int32)
                print(test.shape, test.dtype)

                spec = session.run(model.pred_linear_spec, feed_dict={
                    model.inp_sentences: test
                })

                print('debug', spec)

                spec = spec[0]
                linear_mag_db = inv_normalize_decibel(spec.T, 20, 100)
                linear_mag = decibel_to_magnitude(linear_mag_db)

                print('inversion')
                spec = spectrogram_to_wav(linear_mag,
                                          win_len,
                                          win_hop,
                                          n_fft,
                                          50)

                print('saving')
                save_wav('/tmp/eval_{}.wav'.format(i), spec, 16000, True)
                print('done')


if __name__ == '__main__':
    evaluate(checkpoint_dir='/tmp/tacotron')
