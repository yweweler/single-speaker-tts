import os

import tensorflow as tf

from datasets.dataset import Dataset
from tacotron.input.functions import eval_input_fn
from tacotron.model import Tacotron
from tacotron.params.dataset import dataset_params
from tacotron.params.evaluation import evaluation_params


# Hack to force tensorflow to run on the CPU.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def collect_checkpoint_paths(checkpoint_dir):
    """
    Generates a list of paths to each checkpoint file found in a folder.

    Note:
        - This function assumes, that checkpoint paths were written in relative form.

    Arguments:
        checkpoint_dir (string):
            Path to the models checkpoint directory from which to collect checkpoints.

    Returns:
        paths (:obj:`list` of :obj:`string`):
            List of paths to each checkpoint file.
    """
    listing_file = os.path.join(checkpoint_dir, 'checkpoint')
    lines = []

    # Collect all lines from the checkpoint listing file.
    for line in open(listing_file, 'r'):
        line = line.strip()
        lines.append(line)

    # Discard the first line since it only points to the latest checkpoint.
    lines = lines[1:]

    # Extract the checkpoints path and global step from each line.
    # NOTE: This functions assumes, that all checkpoint paths are relative.
    # all_model_checkpoint_paths: "model.ckpt-<global-step>"

    # Remove "all_model_checkpoint_paths: " from each line.
    lines = [line.replace('all_model_checkpoint_paths: ', '') for line in lines]

    # Remove surrounding quotation marks (" .. ") from each line.
    lines = [line.replace('"', '') for line in lines]

    # Extract the global step from each line.
    # steps = [int(line.split('-', 1)[-1]) for line in lines]

    # Build absolute paths to each checkpoint file.
    paths = [os.path.join(checkpoint_dir, line) for line in lines]

    return paths


def main(_):
    """
    Evaluate a model.
    """
    # Checkpoint folder to load the evaluation checkpoints from.
    checkpoint_load_dir = os.path.join(
        evaluation_params.checkpoint_dir,
        evaluation_params.checkpoint_load_run
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
        session_config=session_config,
        save_summary_steps=evaluation_params.summary_save_steps,
        log_step_count_steps=evaluation_params.performance_log_steps,
        eval_distribute=None
    )

    # Create a model instance.
    model = Tacotron()
    model_fn = model.model_fn

    def __eval_cycle(_checkpoint_file):
        """

        Arguments:
            _checkpoint_file (:obj:`str`):
                Path to the checkpoint file to evaluate.
        """
        # Create a dataset loader.
        eval_dataset = Dataset(dataset_file=dataset_params.dataset_file)
        eval_dataset.load()
        eval_dataset.load_listings()

        # Create a dataset iterator for evaluation.
        input_fn = eval_input_fn(
            dataset_loader=eval_dataset
        )

        # Create a new estimator for evaluation.
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=checkpoint_load_dir,
            config=config,
            params={}
        )

        # Evaluate the model.
        eval_result = estimator.evaluate(input_fn=input_fn, hooks=None,
                                         checkpoint_path=_checkpoint_file)
        print('Evaluation result: {}'.format(eval_result))

    # Check if only the latest or all checkpoints have to be evaluated.
    if evaluation_params.evaluate_all_checkpoints is False:
        # Evaluate the latest checkpoint.
        # NOTE: Passing None as the path to estimator.evaluate will load the latest checkpoint
        # too, but in case the path is invalid or there is no checkpoint the estimator will
        # initialize and evaluate a blank model (Which is not intended).
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_load_dir)

        # Make sure the checkpoint file actually exists on disk.
        assert checkpoint_file is not None, 'No checkpoint was found to load for evaluation!'

        # Evaluate the checkoint.
        __eval_cycle(checkpoint_file)
    else:
        # Get all checkpoints and evaluate them sequentially.
        checkpoint_files = collect_checkpoint_paths(checkpoint_load_dir)
        print("Found #{} checkpoints to evaluate.".format(len(checkpoint_files)))

        # Evaluate all checkoint files.
        for checkpoint_file in checkpoint_files:
            print(checkpoint_file)
            __eval_cycle(checkpoint_file)

            # Reset the graph. Not sure if this is actually needed, but it should not hurt also.
            tf.reset_default_graph()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
