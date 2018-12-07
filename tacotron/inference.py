"""
Use a trained speech synthesis model for prediction.
"""

import os
import re

import tensorflow as tf

from audio.io import save_wav
from datasets.dataset import Dataset
from datasets.utils.processing import prediction_prepare_sentence, synthesize, \
    py_post_process_spectrograms
from tacotron.input.functions import inference_input_fn
from tacotron.model import Tacotron
from tacotron.params.dataset import dataset_params
from tacotron.params.inference import inference_params
from tacotron.params.model import model_params


# Hack to force tensorflow to run on the CPU.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def main(_):
    """
    Predict using a trained model.
    """
    # Check if the required synthesis target folder actually exists.
    if not os.path.isdir(inference_params.synthesis_dir):
        raise NotADirectoryError('The specified synthesis target folder does not exist.')

    # Get the checkpoint folder to load the inference checkpoints from.
    checkpoint_load_dir = os.path.join(
        inference_params.checkpoint_dir,
        inference_params.checkpoint_load_run
    )

    # Configure the session to be created.
    session_config = tf.ConfigProto(
        log_device_placement=False,
        gpu_options=tf.GPUOptions(
            allow_growth=True
        )
    )

    # Configuration for the estimtor.
    config = tf.estimator.RunConfig(
        model_dir=checkpoint_load_dir,
        session_config=session_config
    )

    # Create a model instance.
    model = Tacotron()
    model_fn = model.model_fn

    print('Probing "{}" for checkpoints ...'.format(checkpoint_load_dir))
    if inference_params.checkpoint_file is None:
        # The is no pre-configured checkpoint path to use.
        # Get the path to the latest checkpoint file instead.
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_load_dir)

        # Make sure the latest checkpoint could actually be determined.
        assert checkpoint_file is not None, \
            'Could determine the latest checkpoint in "{}"'.format(checkpoint_load_dir)
    else:
        # Use the pre-configures checkpoint file for prediction.
        checkpoint_file = inference_params.checkpoint_file

    # Make sure the checkpoint to be loaded exists.
    assert os.path.exists(checkpoint_file) is False, \
        'The requested checkpoint file "{}" does not exist.'.format(checkpoint_file)

    print('Loading checkpoint from "{}"'.format(checkpoint_file))

    # Create a dataset loader.
    dataset = Dataset(dataset_file=dataset_params.dataset_file)
    dataset.load()

    # Read all sentences to predict from the synthesis file.
    raw_sentences = []
    with open(inference_params.synthesis_file, 'r') as f_sent:
        for line in f_sent:
            sent = line.replace('\n', '')
            raw_sentences.append(sent)

    print("{} sentences were loaded for inference.".format(len(raw_sentences)))

    # Pre-process the loaded sentences for synthesis.
    whitelist_expression = re.compile(r'[^a-z !?,\-:;()"\']+')
    sentences = [prediction_prepare_sentence(dataset, whitelist_expression, sentence)
                 for sentence in raw_sentences]

    # TODO: Either implement this in the input_fn or remove it entirely and use batching.
    def __build_sentence_generator(_sentences):
        for s in _sentences:
            yield s

    # Create a input function for prediction.
    input_fn = inference_input_fn(
        dataset_loader=dataset,
        sentence_generator=lambda: __build_sentence_generator(sentences)
    )

    # Create a estimator.
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=checkpoint_load_dir,
        config=config,
        params={}
    )

    # Start prediction and read the resulting linear scale magnitude spectrogram.
    predict_result = estimator.predict(input_fn=input_fn,
                                       hooks=None,
                                       predict_keys=['output_linear_spec'],
                                       checkpoint_path=checkpoint_file)

    def synthesis_fn(linear_mag):
        return synthesize(linear_mag, model_params.win_len, model_params.win_hop,
                          model_params.sampling_rate, model_params.n_fft,
                          model_params.magnitude_power, model_params.reconstruction_iterations)

    # Write all generated waveforms to disk.
    for i, (sentence, result) in enumerate(zip(raw_sentences, predict_result)):
        spectrogram = result['output_linear_spec']
        wavs = py_post_process_spectrograms([spectrogram],
                                            dataset,
                                            synthesis_fn,
                                            inference_params.n_synthesis_threads)
        wav = wavs[0]

        # Append ".wav" to the sentence line number to get the filename.
        file_name = '{}.wav'.format(i + 1)

        # Generate the full path under which to save the wav.
        save_path = os.path.join(inference_params.synthesis_dir, file_name)

        # Write the wav to disk.
        save_wav(save_path, wav, model_params.sampling_rate, True)
        print('Saved: "{}" to "{}"'.format(sentence, save_path))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
