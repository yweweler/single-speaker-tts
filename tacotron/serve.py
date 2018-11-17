import tensorflow as tf

from tacotron.input.helpers import py_pre_process_sentences, py_post_process_spectrograms
from tacotron.params.dataset import dataset_params
from tacotron.params.serving import serving_params


# Hack to force tensorflow to run on the CPU.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


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
                batched_sentences = py_pre_process_sentences(raw_sentences, dataset)
                spectrograms = session.run([
                    output_linear_spec
                ],
                    feed_dict={
                        ph_inp_sentences: batched_sentences
                    })

                print(spectrograms)

                wavs = py_post_process_spectrograms(spectrograms)
                print('Generated {} wavefeforms in total'.format(len(wavs)))


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


def main(_):
    def sentence_generator():
        raw_sentences = [
            'Tis a test!',
            'Sentences could be entered using POST requests in the future.'
        ]
        while True:
            yield raw_sentences

    serve(sentence_generator())


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
