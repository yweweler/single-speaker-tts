from multiprocessing.pool import ThreadPool

import numpy as np
import tensorflow as tf

from audio.conversion import inv_normalize_decibel, decibel_to_magnitude, ms_to_samples
from audio.synthesis import spectrogram_to_wav
from tacotron.params.dataset import dataset_params
from tacotron.params.inference import inference_params
from tacotron.params.model import model_params
from tacotron.params.serving import serving_params


def pad_sentence(_sentence, _max_len):
    pad_len = _max_len - len(_sentence)
    pad_token = dataset_params.vocabulary_dict['pad']
    _sentence = np.append(_sentence, [pad_token] * pad_len)

    return _sentence


def pre_process_sentences(_sentences, dataset):
    # Pre-process sentence and convert it into ids.
    id_sequences, sequence_lengths = dataset.process_sentences(_sentences)

    # Get the first sentence.
    sentences = [np.fromstring(id_sequence, dtype=np.int32) for id_sequence in id_sequences]

    # Pad sentence to the same length in order to be able to batch them in a single tensor.
    max_length = max(sequence_lengths)
    sentences = np.array([pad_sentence(sentence, max_length) for sentence in sentences])

    print('sentences', sentences)
    print('sentences.shape', sentences.shape)

    return sentences


def post_process_spectrograms(_spectrograms):
    # Apply Griffin-Lim to all spectrogram's to get the waveforms.
    normalized = list()
    for spectrogram in _spectrograms:
        print('Reverse spectrogram normalization ...', spectrogram.shape)
        linear_mag_db = inv_normalize_decibel(spectrogram.T,
                                              dataset_params.dataset_loader.mel_mag_ref_db,
                                              dataset_params.dataset_loader.mel_mag_max_db)

        linear_mag = decibel_to_magnitude(linear_mag_db)
        normalized.append(linear_mag)

    specs = normalized

    win_len = ms_to_samples(model_params.win_len, model_params.sampling_rate)
    win_hop = ms_to_samples(model_params.win_hop, model_params.sampling_rate)
    n_fft = model_params.n_fft

    def synthesize(linear_mag):
        linear_mag = np.squeeze(linear_mag, -1)
        linear_mag = np.power(linear_mag, model_params.magnitude_power)

        print('Spectrogram inversion ...')
        return spectrogram_to_wav(linear_mag,
                                  win_len,
                                  win_hop,
                                  n_fft,
                                  model_params.reconstruction_iterations)

    # Synthesize waveforms from the spectrograms.
    pool = ThreadPool(inference_params.n_synthesis_threads)
    wavs = pool.map(synthesize, specs)
    pool.close()
    pool.join()

    # # Write all generated waveforms to disk.
    # for i, (sentence, wav) in enumerate(zip(raw_sentences, wavs)):
    #     # Append ".wav" to the sentence line number to get the filename.
    #     file_name = '{}.wav'.format(i + 1)
    #
    #     # Generate the full path under which to save the wav.
    #     save_path = os.path.join(inference_params.synthesis_dir, file_name)
    #
    #     # Write the wav to disk.
    #     # save_wav(save_path, wav, model_params.sampling_rate, True)
    #     print('Saved: "{}"'.format(save_path))

    return wavs


def serve(sentence_generator):
    # Create a dataset loader.
    dataset = dataset_params.dataset_loader(dataset_folder=dataset_params.dataset_folder,
                                            char_dict=dataset_params.vocabulary_dict,
                                            fill_dict=False)

    graph = tf.Graph()
    # Start a session for serving.
    with start_session(graph=graph) as session:
        # Load the exported model into the current session for serving.
        tf.saved_model.loader.load(session,
                                   [tf.saved_model.tag_constants.SERVING],
                                   serving_params.export_dir)

        # Get a handle to the tensor that is filled with the sentences.
        ph_inp_sentences = graph.get_tensor_by_name('ph_inp_sentences:0')

        # The linear spec tensor used as the models output is produced by a dense layer.
        # As such it has no direct name in the exported model (or I am getting something wrong).
        # Hence, the resulting tensor has to be addressed by the bias add operation of the dense
        # layer.
        output_linear_spec = graph.get_tensor_by_name('dense/BiasAdd:0')

        while True:
            # Wait until the sentence generator provides a new set of sentences.
            for raw_sentences in sentence_generator:
                batched_sentences = pre_process_sentences(raw_sentences, dataset)
                spectrograms = session.run([
                    output_linear_spec
                ],
                    feed_dict={
                        ph_inp_sentences: batched_sentences
                    })

                print(spectrograms)

                wavs = post_process_spectrograms(spectrograms)
                print('generated {} wavefeforms in total'.format(len(wavs)))


def start_session(graph=None):
    """
    Creates a session that can be used for training.

    Arguments:
        graph (tf.Graph):
            Default graph to set when creating the session.
            Default is None.

    Returns:
        tf.Session
    """

    session_config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
        )
    )

    session = tf.Session(
        config=session_config,
        graph=graph
    )

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    session.run(init_op)

    return session


if __name__ == '__main__':
    def sentence_generator():
        raw_sentences = [
            'Tis a test!',
            'Sentences could be entered using POST requests in the future.'
        ]
        while True:
            yield raw_sentences


    serve(sentence_generator())
