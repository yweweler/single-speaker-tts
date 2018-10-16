import librosa
import numpy as np
from matplotlib import ticker

from audio.conversion import ms_to_samples, magnitude_to_decibel, \
    normalize_decibel, inv_normalize_decibel, decibel_to_magnitude
from audio.features import linear_scale_spectrogram, mel_scale_spectrogram, calculate_mfccs
from audio.io import load_wav
from audio.visualization import plot_spectrogram, plot_feature_frames
from tacotron.params.model import model_params


def test_db_normalization(wav, hop_len, win_len):
    linear_spec = linear_scale_spectrogram(wav=wav,
                                           n_fft=model_params.n_fft,
                                           hop_length=hop_len,
                                           win_length=win_len).T

    linear_mag = np.abs(linear_spec)
    linear_mag_db = magnitude_to_decibel(linear_mag)

    # Plot the original magnitude spectrogram in dB representation.
    plot_spectrogram(linear_mag_db.T, sr, hop_len, 0.0, 8192.0,
                     'linear',
                     'raw linear scale magnitude spectrogram (dB)')

    linear_mag_db = normalize_decibel(linear_mag_db, 36.50, 100)

    # Plot the normalized magnitude spectrogram in dB representation.
    plot_spectrogram(linear_mag_db.T, sr, hop_len, 0.0, 8192.0,
                     'linear',
                     'normalized linear scale magnitude spectrogram (dB)')

    linear_mag_db = inv_normalize_decibel(linear_mag_db, 36.50, 100)

    # Plot the restored magnitude spectrogram in dB representation.
    plot_spectrogram(linear_mag_db.T, sr, hop_len, 0.0, 8192.0,
                     'linear',
                     'restored linear scale magnitude spectrogram (dB)')


def calculate_mfccs_and_deltas(wav, hop_len, win_len):
    mel_spec = mel_scale_spectrogram(wav,
                                     n_fft=model_params.n_fft,
                                     sampling_rate=model_params.sampling_rate,
                                     hop_length=hop_len,
                                     win_length=win_len,
                                     n_mels=model_params.n_mels,
                                     fmin=model_params.mel_fmin,
                                     fmax=model_params.mel_fmax,
                                     power=2)

    # Convert the Mel-scaled spectrogram to its decibel representation.
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Calculate mfcc features.
    mfccs = calculate_mfccs(mel_spec_db, sr, model_params.n_mfcc)

    plot_spectrogram(mel_spec_db,
                     sampling_rate=model_params.sampling_rate,
                     hop_length=hop_len,
                     fmin=model_params.mel_fmin,
                     fmax=model_params.mel_fmax,
                     y_axis='mel',
                     title='Mel-scale power spectrogram')

    # # Calculate mfcc delta features.
    mfccs_d1 = librosa.feature.delta(mfccs, width=3, order=1)
    # Calculate mfcc delta-delta features.
    mfccs_d2 = librosa.feature.delta(mfccs, width=3, order=2)

    features = np.vstack((mfccs, mfccs_d1, mfccs_d2))

    plot_feature_frames(mfccs, sampling_rate=sr, hop_length=hop_len, title='MFCC')
    plot_feature_frames(mfccs_d1, sampling_rate=sr, hop_length=hop_len, title=r'MFCC-$\Delta$')
    plot_feature_frames(mfccs_d2, sampling_rate=sr, hop_length=hop_len, title=r'MFCC-$\Delta^2$')


def calculate_linear_spec(wav, hop_len, win_len):
    linear_spec = linear_scale_spectrogram(wav,
                                           n_fft=model_params.n_fft,
                                           hop_length=hop_len,
                                           win_length=win_len)

    plot_spectrogram(np.abs(linear_spec),
                     sampling_rate=model_params.sampling_rate,
                     hop_length=hop_len,
                     fmin=model_params.mel_fmin,
                     fmax=model_params.mel_fmax,
                     y_axis='linear',
                     title='Raw Linear-scale spectrogram')

    # Convert the linear-scaled spectrogram to its decibel representation.
    linear_spec_db = librosa.amplitude_to_db(linear_spec, ref=np.max)

    plot_spectrogram(linear_spec_db,
                     sampling_rate=model_params.sampling_rate,
                     hop_length=hop_len,
                     fmin=model_params.mel_fmin,
                     fmax=model_params.mel_fmax,
                     y_axis='linear',
                     title='Linear-scale log spectrogram')


wav_path = '/home/yves-noel/documents/master/thesis/datasets/blizzard_nancy/wav/APDC2-008-03.wav'
# wav_path = '/home/yves-noel/documents/master/thesis/datasets/blizzard_nancy/wav/RURAL-02198.wav'
wav, sr = load_wav(wav_path)

win_len = ms_to_samples(model_params.win_len, sampling_rate=sr)
hop_len = ms_to_samples(model_params.win_hop, sampling_rate=sr)

# --------------------------------------------------------------------------------------------------
# Pitch shifting.
# --------------------------------------------------------------------------------------------------
# wav_ps = pitch_shift(wav, sr, 1/12)
# save_wav('/tmp/ps.wav', wav_ps, sr, True)

# --------------------------------------------------------------------------------------------------
# Silence trimming.
# --------------------------------------------------------------------------------------------------
# wav_trim, indices = trim_silence(wav, 40)
# print(indices)
# save_wav('/tmp/trim.wav', wav_trim, sr, True)

# --------------------------------------------------------------------------------------------------
# Spectrogram normalization and reconstruction.
# --------------------------------------------------------------------------------------------------
# test_db_normalization(wav, hop_len, win_len)

# --------------------------------------------------------------------------------------------------
# Plot waveform
# --------------------------------------------------------------------------------------------------
# plot_waveform(wav, model_params.sampling_rate, title="Blizzard Nancy; APDC2-008-03.wav")

linear_spec = linear_scale_spectrogram(wav, model_params.n_fft, hop_len, win_len).T

mel_spec = mel_scale_spectrogram(wav,
                                 n_fft=model_params.n_fft,
                                 sampling_rate=sr,
                                 n_mels=80,
                                 fmin=0,
                                 fmax=sr // 2,
                                 hop_length=hop_len,
                                 win_length=win_len,
                                 power=1).T

# ==================================================================================================
# Convert the linear spectrogram into decibel representation.
# ==================================================================================================
linear_mag = np.abs(linear_spec)
linear_mag_db = magnitude_to_decibel(linear_mag)

# ==================================================================================================
# Convert the mel spectrogram into decibel representation.
# ==================================================================================================
mel_mag = np.abs(mel_spec)
mel_mag_db = magnitude_to_decibel(mel_mag)



# ==================================================================================================
# Spectrogram plotting.
# ==================================================================================================
from matplotlib import rc

rc('font', **{'family': 'serif',
              'serif': ['Computer Modern'],
              'size': 13.75})
rc('text', usetex=True)

# linear_mag_db = linear_mag_db[int((0.20 * sr) / hop_len):int((1.85 * sr) / hop_len), :]
# fig = plot_spectrogram(linear_mag_db.T, sr, hop_len, 0.0, 8192.0,
#                        'linear', 'APDC2-008-03.wav;      linear scale magnitude spectrogram (dB)',
#                         figsize=((1.0 / 1.35) * (14.0 / 2.54), 7.7 / 2.54))
#
# # DEBUG: Dump plot into a pdf file.
# fig.savefig("/tmp/linear_spectrogram_raw_mag_db.pdf", bbox_inches='tight')

# mel_mag_db = mel_mag_db[int((0.20 * sr) / hop_len):int((1.85 * sr) / hop_len), :]
# fig = plot_spectrogram(mel_mag_db.T, sr, hop_len, 0.0, 8192.0,
#                        'linear', 'APDC2-008-03.wav;      mel scale magnitude spectrogram (dB)',
#                         figsize=((1.0 / 1.35) * (14.0 / 2.54), 7.7 / 2.54))
#
# # DEBUG: Dump plot into a pdf file.
# fig.savefig("/tmp/mel_spectrogram_raw_mag_db.pdf", bbox_inches='tight')

# ==================================================================================================
# Spectrogram normalization.
# ==================================================================================================
print("min mag. (dB): {}".format(np.min(linear_mag_db)))
print("max mag. (dB): {}".format(np.max(linear_mag_db)))

# linear_mag_db = normalize_decibel(linear_mag_db, 36.50, 100)
# print("min norm. mag. (dB): {}".format(np.min(linear_mag_db)))
# print("max norm. mag. (dB): {}".format(np.max(linear_mag_db)))

# Crop the file to the range [4.5s -- 6.2s]
linear_mag_db_raw = linear_mag_db[int((4.5 * sr) / hop_len):int((6.2 * sr) / hop_len), :]
#fraction = ((7.1 / 100) * 4.5) * 2.0
#fraction = 1.0 / 1.35

y_formater = ticker.FuncFormatter(
        lambda x, pos: '{:.0f}'.format(x / 1000.0)
    )

linear_mag_db = linear_mag_db_raw
fig = plot_spectrogram(linear_mag_db.T, sr, hop_len, 0.0, 8192.0,
                       'linear', 'APDC2-008-03.wav; norm. linear scale magnitude spectrogram (dB)',
                       #figsize=(fraction * (1.5 * 14.0 / 2.54), 7.7 / 2.54),
                       figsize=((1.0 / 1.35) * (14.0 / 2.54), 7.7 / 2.54),
                       _formater=y_formater)

# DEBUG: Dump plot into a pdf file.
fig.savefig("/tmp/linear_spectrogram_norm_mag_db.pdf", bbox_inches='tight')

# ==================================================================================================
# Spectrogram raised by to a power.
# ==================================================================================================

linear_mag_db = decibel_to_magnitude(linear_mag_db_raw)
linear_mag_db = np.power(linear_mag_db, 1.4)
linear_mag_db = magnitude_to_decibel(linear_mag_db)

fig = plot_spectrogram(linear_mag_db.T, sr, hop_len, 0.0, 8192.0,
                       'linear',
                       'APDC2-008-03.wav; norm. linear scale magnitude spectrogram (dB) ** 1.3',
                       #figsize=(fraction * (1.5 * 14.0 / 2.54), 7.7 / 2.54),
                       figsize=((1.0 / 1.35) * (14.0 / 2.54), 7.7 / 2.54),
                       _formater=y_formater)

# DEBUG: Dump plot into a pdf file.
fig.savefig("/tmp/linear_spectrogram_norm_mag_db_pow.pdf", bbox_inches='tight')
