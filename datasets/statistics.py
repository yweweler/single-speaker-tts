import matplotlib.pyplot as plt
import numpy as np
import os

from audio.conversion import magnitude_to_decibel, get_duration, ms_to_samples
from audio.features import linear_scale_spectrogram, mel_scale_spectrogram
from audio.io import load_wav
from audio.synthesis import griffin_lim_v2


def collect_duration_statistics(dataset_name, path_listing):
    durations = []

    print("Collecting duration statistics for {} files ...".format(len(path_listing)))
    for path in path_listing:
        # Load the audio file.
        wav, sampling_rate = load_wav(path)
        # Get the duration in seconds.
        duration = get_duration(wav, sampling_rate)
        # Collect durations.
        durations.append(duration)

    durations_sum = sum(durations)
    durations_avg = durations_sum / len(durations)
    durations_min = min(durations)
    durations_max = max(durations)

    print("durations_sum: {} sec.".format(durations_sum))
    print("durations_avg: {} sec.".format(durations_avg))
    print("durations_min: {} sec.".format(durations_min))
    print("durations_max: {} sec.".format(durations_max))

    from matplotlib import rc
    rc('font', **{'family': 'serif',
                  'serif': ['Computer Modern'],
                  'size': 13})
    rc('text', usetex=True)

    # Create a histogram of the individual file durations.
    fig = plt.figure(figsize=(1.5 * 14.0 / 2.54, 7.7 / 2.54), dpi=100)
    plt.hist(durations, bins=100, normed=False, color="#6C8EBF")
    plt.grid(linestyle='dashed')
    plt.xlim([0, 21])
    # plt.title('"{}" file duration distribution'.format(dataset_name))
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Count")
    plt.show()

    # DEBUG: Dump plot into a pdf file.
    fig.savefig("/tmp/durations.pdf", bbox_inches='tight')

    # DEBUG: Dump statistics into a csv file.
    np.savetxt("/tmp/durations.csv", durations, delimiter=",", fmt='%s', header="duration")


def collect_reconstruction_error(path_listing, n_iters):
    mse_errors = []

    n_fft = 2048

    # Window length in ms.
    win_len = 50.0

    # Window stride in ms.
    win_hop = 12.5

    print("Collecting reconstruction statistics for {} files ...".format(len(path_listing)))
    for path in path_listing:
        # Load the audio file.
        wav, sampling_rate = load_wav(path)

        win_len_samples = ms_to_samples(win_len, sampling_rate=sampling_rate)
        win_hop_samples = ms_to_samples(win_hop, sampling_rate=sampling_rate)

        stft = linear_scale_spectrogram(wav,
                                        win_length=win_len_samples,
                                        hop_length=win_hop_samples,
                                        n_fft=n_fft)

        mag = np.abs(stft)
        # mag = np.power(mag, 1.2)

        _, mse = griffin_lim_v2(spectrogram=mag,
                                win_length=win_len_samples,
                                hop_length=win_hop_samples,
                                n_fft=n_fft,
                                n_iter=n_iters)

        # Collect mean-squared errors.
        mse_errors.append(mse)
        # For debugging purposes only.
        # print('"{}" => iters: {}, mse: {}'.format(path, n_iters, mse))

    total_mse = sum(mse_errors) / len(mse_errors)
    print('Dataset MSE with {} iterations: {}'.format(n_iters, total_mse))

    return total_mse


def plot_iterate_reconstruction_error(dataset_name, path_listing, n_iters):
    mse_errors = []
    avg_durations = []

    path_listing = path_listing[:10]

    import time

    _dump_path_mse = '/tmp/griffin_lim_mse.csv'
    _dump_path_mse_durations = '/tmp/griffin_lim_mse_durations.csv'

    if os.path.exists(_dump_path_mse):
        mse_errors = np.loadtxt(_dump_path_mse, delimiter=",", dtype=np.float).tolist()
        avg_durations = np.loadtxt(_dump_path_mse_durations, delimiter=",", dtype=np.float).tolist()
    else:
        for i in range(n_iters):
            start = time.perf_counter()
            mse = collect_reconstruction_error(path_listing, i + 1)
            end = time.perf_counter()

            mse_errors.append(mse)
            avg_durations.append((end - start) / len(path_listing))

            # DEBUG: Dump statistics into a csv file.
            np.savetxt(_dump_path_mse, mse_errors, delimiter=",", fmt='%s', header="mse")
            np.savetxt(_dump_path_mse_durations, avg_durations, delimiter=",", fmt='%s',
                       header="duration")

    mse_errors = mse_errors[:75]
    avg_durations = avg_durations[:75]

    x_iters = np.arange(1, len(mse_errors) + 1)

    from matplotlib import rc
    rc('font', **{'family': 'serif',
                  'serif': ['Computer Modern'],
                  'size': 13})
    rc('text', usetex=True)

    # Create a plot of the MSE related to the number of reconstruction iterations.
    # fig = plt.figure(figsize=((1.5 * 14.0 / 2.54)/1.0, 7.7 / 2.54), dpi=100)
    fig = plt.figure(figsize=((14.0 / 2.54) / 1.35, 7.7 / 2.54), dpi=100)
    plt.plot(x_iters, mse_errors, color="#B85450")#, marker="x")
    # plt.scatter(x_iters, mse_errors, color="#E1D5E7", marker="x")
    plt.grid(linestyle='dashed')
    plt.xlim([1, x_iters[-1]])
    plt.ylim([1e-3, 1.7])
    # plt.title('Griffin-Lim MSE progression')
    plt.xlabel("Iteration")
    plt.ylabel("MSE (dB$^2$)")
    plt.yscale('log')
    plt.show()

    # DEBUG: Dump plot into a pdf file.
    fig.savefig("/tmp/griffin_lim_mse.pdf", bbox_inches='tight')

    # Create a plot of the number of reconstruction iterations related to execution time.
    fig = plt.figure(figsize=((14.0 / 2.54) / 1.25, 7.7 / 2.54), dpi=100)
    plt.plot(x_iters, avg_durations, color="#B85450")#, marker="x")
    plt.grid(linestyle='dashed')
    plt.xlim([1, x_iters[-1]])
    plt.ylim([0, 7])
    # plt.title('Griffin-Lim computation time')
    plt.xlabel("Iteration")
    plt.ylabel("Duration (s)")
    plt.show()

    # DEBUG: Dump plot into a pdf file.
    fig.savefig("/tmp/griffin_lim_mse_durations.pdf", bbox_inches='tight')
