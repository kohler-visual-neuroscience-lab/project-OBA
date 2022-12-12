import mne
import matplotlib.pyplot as plt
import numpy as np
import time
from scipy.stats import ttest_rel


def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
    """Compute SNR spectrum from PSD spectrum using convolution.

    Parameters
    ----------
    psd : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Data object containing PSD values. Works with arrays as produced by
        MNE's PSD functions or channel/trial subsets.
    noise_n_neighbor_freqs : int
        Number of neighboring frequencies used to compute noise level.
        increment by one to add one frequency bin ON BOTH SIDES
    noise_skip_neighbor_freqs : int
        set this >=1 if you want to exclude the immediately neighboring
        frequency bins in noise level calculation

    Returns
    -------
    snr : ndarray, shape ([n_trials, n_channels,] n_frequency_bins)
        Array containing SNR for all epochs, channels, frequency bins.
        NaN for frequencies on the edges, that do not have enough neighbors on
        one side to calculate SNR.
    """
    # Construct a kernel that calculates the mean of the neighboring
    # frequencies
    averaging_kernel = np.concatenate((
        np.ones(noise_n_neighbor_freqs),
        np.zeros(2 * noise_skip_neighbor_freqs + 1),
        np.ones(noise_n_neighbor_freqs)))
    averaging_kernel /= averaging_kernel.sum()

    # Calculate the mean of the neighboring frequencies by convolving with the
    # averaging kernel.
    mean_noise = np.apply_along_axis(
        lambda psd_: np.convolve(psd_, averaging_kernel, mode='valid'),
        axis=-1, arr=psd
    )

    # The mean is not defined on the edges so we will pad it with nas. The
    # padding needs to be done for the last dimension only so we set it to
    # (0, 0) for the other ones.
    edge_width = noise_n_neighbor_freqs + noise_skip_neighbor_freqs
    pad_width = [(0, 0)] * (mean_noise.ndim - 1) + [(edge_width, edge_width)]
    mean_noise = np.pad(
        mean_noise, pad_width=pad_width, constant_values=np.nan
    )

    return psd / mean_noise


# load data
data_path = 'pilot_shaya3_20221125_020929.mff'
raw = mne.io.read_raw_egi(data_path, preload=True)
raw.info['line_freq'] = 60.

# Set montage
montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
raw.set_montage(montage, match_alias=True)
# mne.viz.plot_montage(montage)

# Set common average reference
raw.set_eeg_reference('average', projection=False, verbose=False)

# Apply bandpass filter
raw.filter(l_freq=0.1, h_freq=None, fir_design='firwin', verbose=False)

# Construct epochs
event_id = {'cond1': 1}
events = mne.find_events(raw, stim_channel='TRON')
# plot raw with events on top
# raw.plot(events=events,
#          start=0,
#          duration=30,
#          color='grey',
#          event_color={1: 'r'},
#          block=True)
# times are wrt event times
tmin = 1.5  # in sec
tmax = 7.5  # in sec
epochs = mne.Epochs(raw,
                    events=events,
                    event_id=[event_id['cond1']],
                    tmin=tmin,
                    tmax=tmax,
                    baseline=None,
                    verbose=False)
tmin = tmin
tmax = tmax
fmin = 1.
fmax = 40.
sfreq = epochs.info['sfreq']

spectrum = epochs.compute_psd(
    'welch',
    n_fft=int(sfreq * (tmax - tmin)),
    n_overlap=0, n_per_seg=None,
    tmin=tmin, tmax=tmax,
    fmin=fmin, fmax=fmax,
    window='boxcar',
    verbose=False)

psds, freqs = spectrum.get_data(return_freqs=True)

snrs = snr_spectrum(psds,
                    noise_n_neighbor_freqs=3,
                    noise_skip_neighbor_freqs=1)

# plot PSD and SNR
fig, axes = plt.subplots(2, 1, sharex='all', sharey='none', figsize=(8, 5))
freq_range = range(np.where(np.floor(freqs) == 1.)[0][0],
                   np.where(np.ceil(freqs) == fmax - 1)[0][0])

psds_plot = 10 * np.log10(psds)
psds_mean = psds_plot.mean(axis=(0, 1))[freq_range]
psds_std = psds_plot.std(axis=(0, 1))[freq_range]
axes[0].plot(freqs[freq_range], psds_mean, color='b')
axes[0].fill_between(
    freqs[freq_range], psds_mean - psds_std, psds_mean + psds_std,
    color='b', alpha=.2)
axes[0].set(title="PSD spectrum", ylabel='Power Spectral Density [dB]')

# SNR spectrum
snr_mean = snrs.mean(axis=(0, 1))[freq_range]
snr_std = snrs.std(axis=(0, 1))[freq_range]

axes[1].plot(freqs[freq_range], snr_mean, color='r')
axes[1].fill_between(
    freqs[freq_range], snr_mean - snr_std, snr_mean + snr_std,
    color='r', alpha=.2)
axes[1].set(
    title="SNR spectrum", xlabel='Frequency [Hz]',
    ylabel='SNR', ylim=[-2, 30], xlim=[fmin, fmax])
plt.savefig('PSD_SNR.png')

# find index of frequency bin closest to stimulation frequency
stim_freq1 = 7.5
stim_freq2 = 12
i_bin_1f1 = np.argmin(abs(freqs - stim_freq1))
i_bin_2f1 = np.argmin(abs(freqs - stim_freq1 * 2))
i_bin_3f1 = np.argmin(abs(freqs - stim_freq1 * 3))
i_bin_1f2 = np.argmin(abs(freqs - stim_freq2))
i_bin_2f2 = np.argmin(abs(freqs - stim_freq2 * 2))
i_bin_3f2 = np.argmin(abs(freqs - stim_freq2 * 3))

i_trial_c1 = np.where(epochs.events[:, 2] == event_id['cond1'])[0]

# Define different ROIs
roi_vis = ['E70', 'E74', 'E75', 'E81', 'E82', 'E83']  # visual roi

# Find corresponding indices using mne.pick_types()
picks_roi_vis = mne.pick_types(epochs.info, eeg=True, stim=False,
                               exclude='bads', selection=roi_vis)

# snrs_target1 = snrs[i_trial_c1, :, i_bin_1f1][:, picks_roi_vis]
# snrs_target2 = snrs[i_trial_c1, :, i_bin_1f2][:, picks_roi_vis]
# print("all trials, SNR at {0}".format(stim_freq1))
# print(f'average SNR (occipital ROI): {snrs_target1.mean()}')
# print("all trials, SNR at {0}".format(stim_freq2))
# print(f'average SNR (occipital ROI): {snrs_target2.mean()}')

# get average SNR at xx Hz for ALL channels
ch_average_1f1 = snrs[i_trial_c1, :, i_bin_1f1].mean(axis=0)
ch_average_1f2 = snrs[i_trial_c1, :, i_bin_1f2].mean(axis=0)

# plot SNR topography
_, ax = plt.subplots(1)
mne.viz.plot_topomap(ch_average_1f1,
                     epochs.info,
                     vlim=(1, None),
                     axes=ax)
ax.set_title('1f1')
# plt.savefig('topo_1f1.png')
_, ax = plt.subplots(1)
mne.viz.plot_topomap(ch_average_1f2,
                     epochs.info,
                     vlim=(1, None),
                     axes=ax)
ax.set_title('1f2')
# plt.savefig('topo_1f2.png')
