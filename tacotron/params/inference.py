import tensorflow as tf

# Default hyper-parameters:
inference_params = tf.contrib.training.HParams(
    # Checkpoint folder used for loading the latest checkpoint.
    checkpoint_dir='/thesis/checkpoints/cmu/slt',

    # Run folder to load a checkpoint from the checkpoint folder.
    checkpoint_load_run='train',

    # Direct path to a checkpoint file to restore for inference.
    # If `checkpoint_file` is None, the latest checkpoint will be restored.
    checkpoint_file=None,

    # Run folder to save summaries in the checkpoint folder.
    checkpoint_save_run='inference',

    # The path were to save the inference results.
    synthesis_dir='/thesis/inference/cmu/slt',

    # Path to a file containing sentences to synthesize.
    # On sentence per line is expected.
    synthesis_file='/thesis/inference/sentences.txt',

    # Flag controlling if the alignments should be dumped as .npz files.
    # Dumps are written into `synthesis_dir`/alignments.npz.
    dump_alignments=True,

    # Flag controlling if the linear-scale spectrogram should be dumped as .npz files.
    # Dumps are written into `synthesis_dir`/linear-spectrogram.npz.
    dump_linear_spectrogram=True,

    # The number of process to threads to use for parallel Griffin-Lim reconstruction.
    n_synthesis_threads=6
)
