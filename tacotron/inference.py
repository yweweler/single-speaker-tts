import os

import numpy as np
import tensorflow as tf

from audio.conversion import inv_normalize_decibel, decibel_to_magnitude, ms_to_samples
from audio.io import save_wav
from audio.synthesis import spectrogram_to_wav
from tacotron.model import Tacotron, Mode
from tacotron.params.dataset import dataset_params
from tacotron.params.model import model_params
from tacotron.params.inference import inference_params

# Hack to force tensorflow to run on the CPU.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

tf.logging.set_verbosity(tf.logging.INFO)


def inference(model, sentence):
    """
    Arguments:
        model (Tacotron):
            The Tacotron model instance to be evaluated.

        sentence (string):
            The sentence feed to the network.
    """
    # Checkpoint folder to load the evaluation checkpoint from.
    checkpoint_load_dir = os.path.join(
        inference_params.checkpoint_dir,
        inference_params.checkpoint_load_run
    )

    checkpoint_file = tf.train.latest_checkpoint(checkpoint_load_dir)
    saver = tf.train.Saver()

    # Create the evaluation session.
    session = start_session()

    print('Restoring model...')
    saver.restore(session, checkpoint_file)
    print('Restoring finished')

    # Infer data.
    spectrograms = session.run(
        model.output_linear_spec,
        feed_dict={
            model.inp_sentences: sentence
        })

    spectrogram = spectrograms[0]

    print('Reverse spectrogram normalization ...', spectrogram.shape)
    linear_mag_db = inv_normalize_decibel(spectrogram.T,
                                          dataset_params.mel_mag_ref_db,
                                          dataset_params.mel_mag_max_db)

    linear_mag = decibel_to_magnitude(linear_mag_db)

    win_len = ms_to_samples(model_params.win_len, model_params.sampling_rate)
    win_hop = ms_to_samples(model_params.win_hop, model_params.sampling_rate)
    n_fft = model_params.n_fft

    print('Spectrogram inversion ...')
    spec = spectrogram_to_wav(linear_mag,
                              win_len,
                              win_hop,
                              n_fft,
                              model_params.reconstruction_iterations)

    save_path = '/tmp/inference.wav'
    save_wav(save_path, spec, model_params.sampling_rate, True)
    print('Saved: "{}"'.format(save_path))

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

    return session


if __name__ == '__main__':
    # Create a dataset loader.
    dataset = dataset_params.dataset_loader(dataset_folder=dataset_params.dataset_folder,
                                            char_dict=dataset_params.vocabulary_dict,
                                            fill_dict=False)

    # Pre-process sentence and convert it into ids.
    id_sequences, sequence_lengths = dataset.process_sentences([
        'this is a fucking sentence dude!'
    ])

    # Get the first sentence.
    sentence = np.fromstring(id_sequences[0], dtype=np.int32)

    # Create a batch with only one entry.
    sentence = np.array([sentence], dtype=np.int32)

    # Create batched placeholders for inference.
    placeholders = Tacotron.model_placeholders()

    # Create the Tacotron model.
    tacotron_model = Tacotron(inputs=placeholders, mode=Mode.PREDICT)

    # Train the model.
    inference(tacotron_model, sentence)
