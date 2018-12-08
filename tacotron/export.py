"""
Export a trained model.
"""

import os

import tensorflow as tf

from tacotron.model import Tacotron
from tacotron.params.export import export_params


# Hack to force tensorflow to run on the CPU.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main(_):
    # Checkpoint folder to load the checkpoints to export from.
    checkpoint_load_dir = os.path.join(
        export_params.checkpoint_dir,
        export_params.checkpoint_load_run
    )

    session_config = tf.ConfigProto(
        log_device_placement=False,
        gpu_options=tf.GPUOptions(
            allow_growth=True
        )
    )

    config = tf.estimator.RunConfig(
        model_dir=checkpoint_load_dir,
        session_config=session_config,
        eval_distribute=None
    )

    model = Tacotron()
    model_fn = model.model_fn

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=checkpoint_load_dir,
        config=config,
        params={}
    )

    placeholders = model.model_placeholders()
    # TODO: There are so much input receivers I have no idea which are supposed
    # to be used (Take a look at `ServingInputReceiver`)
    model_path = estimator.export_saved_model(export_params.export_dir,
                                              tf.estimator.export.build_raw_serving_input_receiver_fn(
                                                  {
                                                      'ph_sentences': placeholders['ph_sentences']
                                                  }))

    print('model_path', model_path)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
