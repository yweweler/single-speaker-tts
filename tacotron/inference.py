import os

import tensorflow as tf

from audio.io import save_wav
from tacotron.input.functions import inference_input_fn
from tacotron.input.helpers import py_pre_process_sentences, py_post_process_spectrograms
from tacotron.model import Tacotron
from tacotron.params.dataset import dataset_params
from tacotron.params.inference import inference_params
from tacotron.params.model import model_params

# Hack to force tensorflow to run on the CPU.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

tf.logging.set_verbosity(tf.logging.INFO)


def inference(model, sentences):
    """
    Arguments:
        model (Tacotron):
            The Tacotron model instance to use for inference.

        sentences (:obj:`list` of :obj:`np.ndarray`):
            The padded sentences in id representation to feed to the network.

    Returns:
        spectrograms (:obj:`list` of :obj:`np.ndarray`):
            The generated linear scale magnitude spectrograms.
    """
    # Checkpoint folder to load the inference checkpoint from.
    checkpoint_load_dir = os.path.join(
        inference_params.checkpoint_dir,
        inference_params.checkpoint_load_run
    )

    if inference_params.checkpoint_file is None:
        # Get the path to the latest checkpoint file.
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_load_dir)
    else:
        checkpoint_file = inference_params.checkpoint_file

    saver = tf.train.Saver()

    # Checkpoint folder to save the evaluation summaries into.
    checkpoint_save_dir = os.path.join(
        inference_params.checkpoint_dir,
        inference_params.checkpoint_save_run
    )

    # Prepare the summary writer.
    summary_writer = tf.summary.FileWriter(checkpoint_save_dir, tf.get_default_graph())
    summary_op = model.summary(mode=tf.estimator.ModeKeys.PREDICT)

    # Create the inference session.
    session = start_session()

    print('Restoring model...')
    saver.restore(session, checkpoint_file)
    print('Restoring finished')

    # Infer data.
    summary, spectrograms = session.run(
        # TODO: implement automatic stopping after a certain amount of silence was generated.
        # Then we could set max_iterations much higher and only use it as a worst case fallback
        # when the network does not stop by itself.
        [
            summary_op,
            model.output_linear_spec
        ],
        feed_dict={
            model.inp_sentences: sentences
        })

    # Write the summary statistics.
    inference_summary = tf.Summary()
    inference_summary.ParseFromString(summary)
    summary_writer.add_summary(inference_summary)

    session.close()

    return spectrograms


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


def main(_):
    # Before we start doing anything we check if the required target folder actually exists.
    if not os.path.isdir(inference_params.synthesis_dir):
        raise NotADirectoryError('The specified synthesis target folder does not exist.')

    # Checkpoint folder to load the inference checkpoints from.
    checkpoint_load_dir = os.path.join(
        inference_params.checkpoint_dir,
        inference_params.checkpoint_load_run
    )

    session_config = tf.ConfigProto(
        log_device_placement=False,
        gpu_options=tf.GPUOptions(
            allow_growth=True
        )
    )

    config = tf.estimator.RunConfig(
        model_dir=checkpoint_load_dir,
        session_config=session_config
    )

    model = Tacotron()
    model_fn = model.model_fn

    print('Probing "{}" for checkpoints ...'.format(checkpoint_load_dir))
    if inference_params.checkpoint_file is None:
        # Get the path to the latest checkpoint file.
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_load_dir)
        assert checkpoint_file is not None, \
            'Could determine the latest checkpoint in "{}"'.format(checkpoint_load_dir)
    else:
        checkpoint_file = inference_params.checkpoint_file

    assert os.path.exists(checkpoint_file) is False, \
        'The requested checkpoint file "{}" does not exist.'.format(checkpoint_file)

    print('Loading checkpoint from "{}"'.format(checkpoint_file))

    # Create a dataset loader.
    dataset = dataset_params.dataset_loader(dataset_folder=dataset_params.dataset_folder,
                                            char_dict=dataset_params.vocabulary_dict,
                                            fill_dict=False)

    raw_sentences = []
    with open(inference_params.synthesis_file, 'r') as f_sent:
        for line in f_sent:
            sent = line.replace('\n', '')
            raw_sentences.append(sent)

    print("{} sentences were loaded for inference.".format(len(raw_sentences)))

    sentences = py_pre_process_sentences(raw_sentences, dataset)

    def __build_sentence_generator(*args):
        _sentences = args[0]
        for s in _sentences:
            yield s

    input_fn = inference_input_fn(
        dataset_loader=dataset,
        sentence_generator=__build_sentence_generator,
        sentences=sentences
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=checkpoint_load_dir,
        config=config,
        params={}
    )

    # Start prediction.
    print('calling: estimator.predict')
    predict_result = estimator.predict(input_fn=input_fn,
                                       hooks=None,
                                       predict_keys=['output_linear_spec'],
                                       checkpoint_path=checkpoint_file)

    print('Prediction result: {}'.format(predict_result))

    # Write all generated waveforms to disk.
    for i, (sentence, result) in enumerate(zip(raw_sentences, predict_result)):
        spectrogram = result['output_linear_spec']
        wavs = py_post_process_spectrograms([spectrogram])
        wav = wavs[0]

        # Append ".wav" to the sentence line number to get the filename.
        file_name = '{}.wav'.format(i + 1)

        # Generate the full path under which to save the wav.
        save_path = os.path.join(inference_params.synthesis_dir, file_name)

        # Write the wav to disk.
        save_wav(save_path, wav, model_params.sampling_rate, True)
        print('Saved: "{}"'.format(save_path))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
