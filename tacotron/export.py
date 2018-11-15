import os

import tensorflow as tf

from tacotron.model import Tacotron, Mode
from tacotron.params.export import export_params

# Hack to force tensorflow to run on the CPU.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

tf.logging.set_verbosity(tf.logging.INFO)


def export(model):
    """
    Arguments:
        model (Tacotron):
            The Tacotron model instance to use for export.
    """
    # Checkpoint folder to load the export checkpoint from.
    checkpoint_load_dir = os.path.join(
        export_params.checkpoint_dir,
        export_params.checkpoint_load_run
    )

    if export_params.checkpoint_file is None:
        # Get the path to the latest checkpoint file.
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_load_dir)
    else:
        checkpoint_file = export_params.checkpoint_file

    with tf.device('/cpu:0'):
        saver = tf.train.Saver()

        # Create the export session.
        session = start_session()

        print('Restoring model...')
        saver.restore(session, checkpoint_file)
        print('Restoring finished')

        # =========================================================================

        # Create SavedModelBuilder class.
        # Defines where the model will be exported and with which version number.
        export_path_base = export_params.export_dir
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(export_params.export_version))
        )

        print('Exporting trained model as version {} to "{}"'
              .format(export_params.export_version, export_path))

        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        # Creates the TensorInfo protobuf objects that encapsulates the input/output tensors.
        tensor_info_input = tf.saved_model.utils.build_tensor_info(model.inp_sentences)
        tensor_info_output = tf.saved_model.utils.build_tensor_info(model.output_linear_spec)

        # Defines the model signatures, uses the TF Predict API.
        # It receives an sentence and outputs the linear spectrogram.
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'ph_inp_sentences': tensor_info_input},
                outputs={'output_linear_spec': tensor_info_output},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

        builder.add_meta_graph_and_variables(
            session, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_spectrogram':
                    prediction_signature,
            })

        # Export the model
        builder.save(as_text=True)

        print('Done exporting!')

        session.close()


def start_session():
    """
    Creates a session that can be used for exporting the model.

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
    # Create empty batched placeholders for export.
    placeholders = Tacotron.model_placeholders()

    # Create the Tacotron model.
    tacotron_model = Tacotron(inputs=placeholders, mode=Mode.PREDICT)

    # Export the model.
    export(tacotron_model)
