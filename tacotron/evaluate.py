import os

import tensorflow as tf

from tacotron.input.functions import eval_input_fn
from tacotron.input.helpers import placeholders_from_dataset_iter
from tacotron.model import Tacotron, Mode
from tacotron.params.dataset import dataset_params
from tacotron.params.evaluation import evaluation_params


# Hack to force tensorflow to run on the CPU.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def evaluate(model, checkpoint_file):
    """
    Evaluates a Tacotron model.

    Arguments:
        model (Tacotron):
            The Tacotron model instance to be evaluated.

        checkpoint_file (string):
            Absolute path to the checkpoint file to be evaluated.
    """
    # Get the models loss function.
    loss_op = model.get_loss_op()

    # Checkpoint folder to load the evaluation checkpoint from.
    checkpoint_load_dir = os.path.join(
        evaluation_params.checkpoint_dir,
        evaluation_params.checkpoint_load_run
    )

    print('checkpoint_file', checkpoint_file)

    # Checkpoint folder to save the evaluation summaries into.
    checkpoint_save_dir = os.path.join(
        evaluation_params.checkpoint_dir,
        evaluation_params.checkpoint_save_run
    )

    # Get the checkpoints global step from the checkpoints file name.
    global_step = int(checkpoint_file.split('-')[-1])
    print('[checkpoint_file] step: {}, file: "{}"'.format(global_step, checkpoint_file))

    saver = tf.train.Saver()

    summary_writer = tf.summary.FileWriter(checkpoint_save_dir, tf.get_default_graph(),
                                           flush_secs=10)
    summary_op = model.summary(mode=tf.estimator.ModeKeys.EVAL)

    # ========================================================

    session_config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
        )
    )

    with tf.Session(config=session_config) as session:

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        session.run(init_op)

        tf.train.start_queue_runners(sess=session)

        # ========================================================

        # Create the evaluation session.
        # with start_session() as session:
        print('Restoring model...')
        saver.restore(session, checkpoint_file)
        print('Restoring finished')

        sum_loss = 0
        sum_loss_decoder = 0
        sum_loss_post_processing = 0

        batch_count = 0
        summary = None

        # Start evaluation.
        while True:
            try:
                # Evaluate loss functions for the current batch.
                summary, loss, loss_decoder, loss_post_processing = session.run([
                    summary_op,
                    model.loss_op,
                    model.loss_op_decoder,
                    model.loss_op_post_processing,
                ])

                # Accumulate loss values.
                sum_loss += loss
                sum_loss_decoder += loss_decoder
                sum_loss_post_processing += loss_post_processing

                # Increment batch counter.
                batch_count += 1

            except tf.errors.OutOfRangeError:
                if batch_count == 0:
                    raise Exception("Error: No batches were processed!")
                    exit(1)
                break

        avg_loss = sum_loss / batch_count
        avg_loss_decoder = sum_loss_decoder / batch_count
        avg_loss_post_processing = sum_loss_post_processing / batch_count

        # Create evaluation summaries.
        eval_summary = tf.Summary()

        eval_summary.ParseFromString(summary)
        eval_summary.value.add(tag='loss/loss', simple_value=avg_loss)
        eval_summary.value.add(tag='loss/loss_decoder', simple_value=avg_loss_decoder)
        eval_summary.value.add(tag='loss/loss_post_processing',
                               simple_value=avg_loss_post_processing)

        summary_writer.add_summary(eval_summary, global_step=global_step)


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

    tf.train.start_queue_runners(sess=session)

    return session


def collect_checkpoint_paths(checkpoint_dir):
    """
    Generates a list of paths to each checkpoint file found in a folder.

    Note:
        - This function assumes, that checkpoint paths were written in relative.

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
    # Checkpoint folder to load the evaluation checkpoints from.
    checkpoint_load_dir = os.path.join(
        evaluation_params.checkpoint_dir,
        evaluation_params.checkpoint_load_run
    )

    def __eval_cycle(_checkpoint_file):
        # Create a dataset loader.
        eval_dataset = dataset_params.dataset_loader(dataset_folder=dataset_params.dataset_folder,
                                                     char_dict=dataset_params.vocabulary_dict,
                                                     fill_dict=False)

        # Create a dataset iterator for evaluation.
        dataset_iter = eval_input_fn(
            dataset_loader=eval_dataset
        )

        # Create placeholders from the dataset iterator.
        placeholders = placeholders_from_dataset_iter(dataset_iter)

        # Create the Tacotron model.
        tacotron_model = Tacotron(inputs=placeholders, mode=Mode.EVAL)

        # Evaluate the model.
        evaluate(tacotron_model, _checkpoint_file)
        tf.reset_default_graph()

    if evaluation_params.evaluate_all_checkpoints is False:
        # Get the latest checkpoint for evaluation.
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_load_dir)
        __eval_cycle(checkpoint_file)
    else:
        # Get all checkpoints and evaluate the sequentially.
        checkpoint_files = collect_checkpoint_paths(checkpoint_load_dir)
        print("Found #{} checkpoints to evaluate.".format(len(checkpoint_files)))
        for checkpoint_file in checkpoint_files:
            print(checkpoint_file)
            __eval_cycle(checkpoint_file)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
