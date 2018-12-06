def inputs_from_dataset_iter(dataset_iter):
    # get inputs from the dataset iterator.
    ph_sentences, ph_sentence_lengths, ph_mel_specs, ph_lin_specs, ph_time_frames = \
        dataset_iter.get_next()

    inputs = {
        'ph_sentences': ph_sentences,
        'ph_sentence_length': ph_sentence_lengths,
        'ph_mel_specs': ph_mel_specs,
        'ph_lin_specs': ph_lin_specs,
        'ph_time_frames': ph_time_frames
    }

    return inputs
