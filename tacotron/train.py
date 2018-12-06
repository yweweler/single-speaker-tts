import os

import tensorflow as tf

from datasets.dataset import Dataset
from tacotron.input.functions import train_input_fn
from tacotron.model import Tacotron
from tacotron.params.dataset import dataset_params
from tacotron.params.training import training_params


# Hack to force tensorflow to run on the CPU.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def main(_):
    """
    Train a model.
    """
    # Create a dataset loader.
    train_dataset = Dataset(dataset_file=dataset_params.dataset_file)
    train_dataset.load()

    # Get the folder to load checkpoints from.
    checkpoint_dir = os.path.join(training_params.checkpoint_dir, training_params.checkpoint_run)

    # Configure the session to be created.
    session_config = tf.ConfigProto(
        log_device_placement=False,
        gpu_options=tf.GPUOptions(
            allow_growth=True
        )
    )

    # NOTE: During training an estimator may add the following hooks on its own:
    # (NanTensorHook, LoggingTensorHook, CheckpointSaverHook).
    # A `NanTensorHook` is always created.
    # A `LoggingTensorHook` is created if `log_step_count_steps` is set.
    # A `CheckpointSaverHook` is not created if an existing hook is found in `training_hooks`.
    # If multiple `CheckpointSaverHook` objects are found only the first is used (This behaviour
    # is not very obvious as it does not output a warning).
    config = tf.estimator.RunConfig(
        model_dir=checkpoint_dir,
        session_config=session_config,
        save_summary_steps=training_params.summary_save_steps,
        save_checkpoints_steps=training_params.checkpoint_save_steps,
        keep_checkpoint_max=training_params.checkpoints_to_keep,
        log_step_count_steps=training_params.performance_log_steps,
        train_distribute=None
    )

    # Create a model instance.
    model = Tacotron()
    model_fn = model.model_fn

    # Create an estimator.
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=checkpoint_dir,
        config=config,
        params={}
    )

    # Create a dataset iterator for training.
    input_fn = train_input_fn(
        dataset_loader=train_dataset
    )

    # Train the model.
    estimator.train(input_fn=input_fn, hooks=None, steps=None, max_steps=None)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
