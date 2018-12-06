import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc

from tacotron.input.helpers import py_pre_process_sentences
from tacotron.params.dataset import dataset_params
from tacotron.params.serving import serving_params


# Hack to force tensorflow to run on the CPU.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def main(_):
    # Export folder to load the exported model from.
    checkpoint_load_dir = os.path.join(
        serving_params.export_dir,
        str(serving_params.export_version)
    )

    session_config = tf.ConfigProto(
        log_device_placement=False,
        gpu_options=tf.GPUOptions(
            allow_growth=True
        )
    )

    predict_fn = tfc.predictor.from_saved_model(export_dir=checkpoint_load_dir,
                                                config=session_config)
    sentence = np.zeros(shape=(28), dtype=np.int32)
    prediction = predict_fn({
        "ph_sentences": [sentence]
    })

    print('Generated spectrogram {}'.format(prediction['output_linear_spec'].shape))

    # TODO: The interface has changed, rewrite call.
    wav = py_post_process_spectrograms(prediction['output_linear_spec'])
    print('Generated waveform')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
