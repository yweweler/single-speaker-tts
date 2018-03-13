from time import sleep

import tensorflow as tf
from tqdm import trange

from tacotron.hparams import hparams
from tacotron.model import Tacotron


def train_data():
    raise NotImplementedError()


def train(checkpoint_dir):
    n_epochs = 2
    n_batches = 10

    # Checkpoint every 10 minutes.
    checkpoint_save_secs = 60 * 10

    # Save summary every 10 steps.
    summary_save_steps = 10

    model = Tacotron(hparams=hparams)
    loss_op = model.loss()
    summary_op = model.summary()

    # TODO: Not sure what the exact benefit of a scaffold is since it does not hold very much data.
    session_scaffold = tf.train.Scaffold(
        init_op=tf.global_variables_initializer(),
        summary_op=summary_op
    )

    saver_hook = tf.train.CheckpointSaverHook(
        checkpoint_dir=checkpoint_dir,
        save_secs=checkpoint_save_secs,
        scaffold=session_scaffold
    )

    summary_hook = tf.train.SummarySaverHook(
        output_dir=checkpoint_dir,
        save_steps=summary_save_steps,
        scaffold=session_scaffold
    )

    nan_hook = tf.train.NanTensorHook(
        loss_tensor=loss_op,
        fail_on_nan_loss=True
    )

    session_config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=True,
        )
    )

    session = tf.train.SingularMonitoredSession(hooks=[saver_hook, summary_hook, nan_hook],
                                                scaffold=session_scaffold,
                                                config=session_config,
                                                checkpoint_dir=checkpoint_dir)

    # TODO: Is this necessary? I guess the MonitoredSession does this itself using `session_scaffold.init_op`.
    # session.run(tf.global_variables_initializer())

    # TODO: Is this necessary? I guess the MonitoredSession will load the network from a checkpoint itself.
    # model.load(checkpoint_dir)

    # TODO: Iterate epochs.
    # TODO: Feed batches to the network and update gradients.
    for epoch in range(n_epochs):
        batches = trange(n_batches, dynamic_ncols=True)
        for batch in batches:
            batches.set_description('Epoch {}'.format(epoch + 1))
            # batches.set_postfix(loss=0.0)
            sleep(0.15)

    session.close()


if __name__ == '__main__':
    train(checkpoint_dir='/tmp/tacotron')
