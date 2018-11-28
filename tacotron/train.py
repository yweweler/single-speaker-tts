import os

import tensorflow as tf

from tacotron.input.functions import train_input_fn
from tacotron.model import Tacotron
from tacotron.params.dataset import dataset_params
from tacotron.params.training import training_params


# Hack to force tensorflow to run on the CPU.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def main(_):
    # Create a dataset loader.
    train_dataset = dataset_params.dataset_loader(dataset_folder=dataset_params.dataset_folder,
                                                  char_dict=dataset_params.vocabulary_dict,
                                                  fill_dict=False)

    checkpoint_dir = os.path.join(training_params.checkpoint_dir, training_params.checkpoint_run)

    session_config = tf.ConfigProto(
        log_device_placement=True,
        gpu_options=tf.GPUOptions(
            allow_growth=True
        )
    )

    config = tf.estimator.RunConfig(
        model_dir=checkpoint_dir,
        session_config=session_config,
        save_summary_steps=training_params.summary_save_steps,
        save_checkpoints_steps=training_params.checkpoint_save_steps,
        keep_checkpoint_max=training_params.checkpoints_to_keep,
        log_step_count_steps=training_params.performance_log_steps,
        train_distribute=None
    )

    model = Tacotron()
    model_fn = model.model_fn

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=checkpoint_dir,
        config=config,
        params={}
    )

    # Create a dataset iterator for training.
    # TODO: Rewrite the dataset helpers to make it easier to handle train and eval portions.
    input_fn = train_input_fn(
        dataset_loader=train_dataset
    )

    # Train the model.
    estimator.train(input_fn=input_fn, hooks=None, steps=None, max_steps=None)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
