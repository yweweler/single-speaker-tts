import os

import tensorflow as tf

from tacotron.input.functions import train_input_fn
from tacotron.input.helpers import placeholders_from_dataset_iter
from tacotron.model import Tacotron
from tacotron.params.dataset import dataset_params
from tacotron.params.training import training_params


# Hack to force tensorflow to run on the CPU.
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def train(model):
    """
    Trains a Tacotron model.

    Arguments:
        model (Tacotron):
            The Tacotron model instance to be trained.
    """
    # NOTE: The global step has to be created before the optimizer is created.
    global_step = tf.train.create_global_step()

    # Get the models loss function.
    loss_op = model.get_loss_op()

    with tf.name_scope('optimizer'):
        # Let the learning rate decay exponentially.
        learning_rate = tf.train.exponential_decay(
            learning_rate=training_params.lr,
            global_step=global_step,
            decay_steps=training_params.lr_decay_steps,
            decay_rate=training_params.lr_decay_rate,
            staircase=training_params.lr_staircase)

        # Force decrease to stop at a minimal learning rate.
        learning_rate = tf.maximum(learning_rate, training_params.minimum_lr)

        # Add a learning rate summary.
        tf.summary.scalar('lr', learning_rate)

        # Create a optimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Apply gradient clipping by global norm.
        gradients, variables = zip(*optimizer.compute_gradients(loss_op))
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, training_params.gradient_clip_norm)

        # Add dependency on UPDATE_OPS; otherwise batch normalization won't work correctly.
        # See: https://github.com/tensorflow/tensorflow/issues/1122
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimize = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step)

    # Create the training session.
    session = start_session(loss_op=loss_op,
                            summary_op=model.summary(mode=tf.estimator.ModeKeys.TRAIN))

    # Start training.
    while not session.should_stop():
        try:
            session.run([global_step, loss_op, optimize])
        except tf.errors.OutOfRangeError:
            break

    session.close()


# TODO: Port the session functionality to the estimator.
def start_session(loss_op, summary_op):
    """
    Creates a session that can be used for training.

    Arguments:
        loss_op (tf.Tensor):

        summary_op (tf.Tensor):
            A tensor of type `string` containing the serialized `Summary` protocol
            buffer containing all merged model summaries.

    Returns:
        tf.train.SingularMonitoredSession
    """
    checkpoint_dir = os.path.join(training_params.checkpoint_dir, training_params.checkpoint_run)

    saver = tf.train.Saver(
        # NOTE: CUDNN RNNs do not support distributed saving of parameters.
        sharded=False,
        allow_empty=True,
        max_to_keep=training_params.checkpoints_to_keep,
        save_relative_paths=True
    )

    saver_hook = tf.train.CheckpointSaverHook(
        checkpoint_dir=checkpoint_dir,
        # save_secs=training_params.checkpoint_save_secs,
        save_steps=training_params.checkpoint_save_steps,
        saver=saver
    )

    summary_hook = tf.train.SummarySaverHook(
        output_dir=checkpoint_dir,
        save_steps=training_params.summary_save_steps,
        summary_op=summary_op
    )

    nan_hook = tf.train.NanTensorHook(
        loss_tensor=loss_op,
        fail_on_nan_loss=True
    )

    counter_hook = tf.train.StepCounterHook(
        output_dir=checkpoint_dir,
        every_n_steps=training_params.performance_log_steps
    )

    session_config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
        )
    )

    session = tf.train.SingularMonitoredSession(hooks=[
        saver_hook,
        summary_hook,
        nan_hook,
        counter_hook],
        # Note: When using a monitored session in combination with CUDNN RNNs this needs to be
        # set otherwise the CUDNN RNN does not find a default device to collect variables for
        # saving.
        scaffold=tf.train.Scaffold(saver=saver),
        config=session_config,
        checkpoint_dir=checkpoint_dir)

    tf.train.start_queue_runners(sess=session)

    return session


def main(_):
    # Create a dataset loader.
    train_dataset = dataset_params.dataset_loader(dataset_folder=dataset_params.dataset_folder,
                                                  char_dict=dataset_params.vocabulary_dict,
                                                  fill_dict=False)

    # Create a dataset iterator for training.
    input_fn = train_input_fn(
        dataset_loader=train_dataset
    )

    checkpoint_dir = os.path.join(training_params.checkpoint_dir, training_params.checkpoint_run)

    session_config = tf.ConfigProto(
        log_device_placement=True,
        gpu_options=tf.GPUOptions(
            allow_growth=True
        )
    )

    config = tf.estimator.RunConfig(
        model_dir=checkpoint_dir,
        session_config=session_config
    )

    model = Tacotron(training_summary=training_params.write_summary)
    model_fn = model.model_fn

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=checkpoint_dir,
        config=config
    )

    # Train the model.
    estimator.train(input_fn=input_fn, hooks=None, steps=None, max_steps=None)

    # # Create placeholders from the dataset iterator.
    # placeholders = placeholders_from_dataset_iter(dataset_iter)
    #
    # # Create the Tacotron model.
    # tacotron_model = Tacotron(inputs=placeholders, mode=Mode.TRAIN,
    #                           training_summary=training_params.write_summary)
    #
    # # Train the model.
    # train(tacotron_model)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
