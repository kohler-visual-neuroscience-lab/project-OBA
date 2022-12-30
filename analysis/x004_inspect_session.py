import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cleanplot as cp

"""
Mohammad Shams <mShamsResearch@gmail.com> Dec 28, 2022

- Inspects the recorded behavioral and EEG data of a single session.
- Generates two figures: behavioral analysis, SSVEP analysis

"""


# /// functions
def snr_spectrum(psd, noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=1):
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


# ----------------------------------------------------------------------------

# /// LOAD DATA ///

eeg_file = '0002_20221222_025047.mff'
beh_file = '0002_20221222_145047.json'
# set the full path to the raw data
eeg_path = os.path.join('..', 'data', 'rawData', 'test_exp01', eeg_file)
beh_path = os.path.join('..', 'data', 'rawData', 'test_exp01', beh_file)
eeg = mne.io.read_raw_egi(eeg_path, preload=True)
beh_data = pd.read_json(beh_path)
# extract subject's ID
sub_id = beh_file[:4]
# ----------------------------------------------------------------------------

# /// SETUP BEHAVIORAL DATA ///

# extract number of trials
n_trials = beh_data.shape[0]
# convert tilt magnitudes to degrees
tilt_angle = (beh_data['tilt_magnitude'].values + 1) / 10
# ----------------------------------------------------------------------------

# /// SETUP EEG DATA ///

eeg.info['line_freq'] = 60.
# set montage
montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
eeg.set_montage(montage, match_alias=True)
# set common average reference
eeg.set_eeg_reference('average', projection=False, verbose=False)
# apply bandpass filter
eeg.filter(l_freq=0.1, h_freq=None, fir_design='firwin', verbose=False)

# /// extract events
events = mne.find_events(eeg, stim_channel=['CND1', 'CND2', 'CND3', 'CND4'])
event_id = eeg.event_id

# /// reshape data into epochs
tmin = 0  # in sec wrt event times
tmax = 10  # in sec wrt event times
epochs = mne.Epochs(eeg, events=events,
                    event_id=[event_id['CND1'], event_id['CND2'],
                              event_id['CND3'], event_id['CND4']],
                    tmin=tmin, tmax=tmax, baseline=None, verbose=False)

# /// calculate signal-to-noise ratio (SNR) and power spectral density (PSD)
tmin = tmin
tmax = tmax
fmin = 1.
fmax = 30.
sampling_freq = epochs.info['sfreq']
# calculate PSD
spectrum = epochs.compute_psd('welch',
                              n_fft=int(sampling_freq * (tmax - tmin)),
                              n_overlap=0, n_per_seg=None, tmin=tmin,
                              tmax=tmax, fmin=fmin, fmax=fmax, window='boxcar',
                              verbose=False)
psds, freqs = spectrum.get_data(return_freqs=True)
# calculate SNR
snrs = snr_spectrum(psds, noise_n_neighbor_freqs=3,
                    noise_skip_neighbor_freqs=1)
# define a freq range
freq_range = range(np.where(np.floor(freqs) == 1.)[0][0],
                   np.where(np.ceil(freqs) == fmax - 1)[0][0])
# average PSD across all trials and all channels within the desired freq range
psds_plot = 10 * np.log10(psds)
psds_mean = psds_plot.mean(axis=(0, 1))[freq_range]
psds_std = psds_plot.std(axis=(0, 1))[freq_range]
# average SNR across all trials and all channels within the desired freq range
snr_mean = snrs.mean(axis=(0, 1))[freq_range]
snr_std = snrs.std(axis=(0, 1))[freq_range]

# /// prepare data for topography maps
# index the closest frequency bin to stimulation frequency
stim_freq1 = 7.5
i_bin_1f1 = np.argmin(abs(freqs - stim_freq1 * 1))
i_bin_2f1 = np.argmin(abs(freqs - stim_freq1 * 2))
i_bin_3f1 = np.argmin(abs(freqs - stim_freq1 * 3))
stim_freq2 = 12
i_bin_1f2 = np.argmin(abs(freqs - stim_freq2 * 1))
i_bin_2f2 = np.argmin(abs(freqs - stim_freq2 * 2))
i_bin_3f2 = np.argmin(abs(freqs - stim_freq2 * 3))
# index first harmonics of both stimulus frequencies
# NOTE: the dimensions of snrs are: events/trials x channels x freq_bins
snrs_1f1 = snrs[:, :, i_bin_1f1]
snrs_1f2 = snrs[:, :, i_bin_1f2]
# average across events/trials
snrs_1f1_avg = snrs_1f1.mean(axis=0)
snrs_1f2_avg = snrs_1f2.mean(axis=0)
# ----------------------------------------------------------------------------

# +++ TEST +++

# make sure that number of events (in eeg file) matches the number of trials
# (in beh file)
assert events.shape[0] == beh_data.shape[0]
# ----------------------------------------------------------------------------

#  @@@ PLOT BEHAVIORAL ANALYSES @@@

_, axs = plt.subplots(4, 2, figsize=(10, 8), width_ratios=[3, 1])
cp.prep4ai()

# events as a function of trials
x_evnt1 = np.nonzero(events[:, 2] == 1)
axs[0, 0].plot(x_evnt1, 1 * np.ones(len(x_evnt1)), 'o',
               markerfacecolor='r', markeredgecolor='none')
x_evnt2 = np.nonzero(events[:, 2] == 2)
axs[0, 0].plot(x_evnt2, 2 * np.ones(len(x_evnt2)), 'o',
               markerfacecolor='g', markeredgecolor='none')
x_evnt3 = np.nonzero(events[:, 2] == 3)
axs[0, 0].plot(x_evnt3, 3 * np.ones(len(x_evnt3)), 'o',
               markerfacecolor='b', markeredgecolor='none')
x_evnt4 = np.nonzero(events[:, 2] == 4)
axs[0, 0].plot(x_evnt4, 4 * np.ones(len(x_evnt4)), 'o',
               markerfacecolor='m', markeredgecolor='none')
axs[0, 0].set(yticks=[1, 2, 3, 4],
              yticklabels=['CND1', 'CND2', 'CND3', 'CND4'],
              ylabel='Events', xlim=[0 - 2, n_trials], ylim=[1 - .5, 4.5])

# performances over session
axs[1, 0].plot(beh_data['cummulative_performance'], color='blue')
axs[1, 0].plot(beh_data['running_performance'], color='tomato')
axs[1, 0].set(ylabel='Performance [%]',
              xlim=[0 - 2, n_trials], ylim=[0 - 5, 100])
cp.trim_axes(axs[1, 0])

# RT over session
axs[2, 0].plot(beh_data['avg_rt'], 'o', markerfacecolor='k',
               markeredgecolor='none')
axs[2, 0].set_yticks(range(0, 1000 + 1, 250))
axs[2, 0].set(ylabel='RT [ms]',
              xlim=[0 - 2, n_trials], ylim=[0 - 50, 1000])
cp.trim_axes(axs[2, 0])

# tilt angles over session
axs[3, 0].plot(tilt_angle, color='k')
axs[3, 0].set(xlabel='Trials', ylabel='Tilt angle [deg]',
              xlim=[0 - 2, n_trials], ylim=[0 - .3, 6])
cp.trim_axes(axs[3, 0])

#  leave this subplot empty
axs[0, 1].axis('off')

#  leave this subplot empty
axs[1, 1].axis('off')

# RT histogram
hist_bins = range(0, 1000, 100)
axs[2, 1].hist(beh_data['avg_rt'], facecolor='k', bins=hist_bins)
axs[2, 1].set_xticks(range(0, 1000 + 1, 250))
axs[2, 1].set(xticks=range(0, 1000 + 1, 250),
              xlabel='RT [ms]', ylabel='Count',
              xlim=[0 - 30, 1000], ylim=[0 - .5, 30])
cp.trim_axes(axs[2, 1])

#  leave this subplot empty
axs[3, 1].axis('off')

plt.show()
# ----------------------------------------------------------------------------

# @@@ PLOT EEG DATA ANALYSES @@@

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle(f'Subject ID: {sub_id} — All channels')
cp.prep4ai()

# PSD spectrum
axes[0, 0].plot(freqs[freq_range], psds_mean, color='k')
axes[0, 0].fill_between(
    freqs[freq_range], psds_mean - psds_std, psds_mean + psds_std,
    color='k', alpha=.2)
axes[0, 0].set(ylabel='PSD [dB]', xlim=[fmin, fmax])
cp.trim_axes(axes[0, 0])

# SNR spectrum
axes[1, 0].plot(freqs[freq_range], snr_mean, color='k')
axes[1, 0].fill_between(
    freqs[freq_range], snr_mean - snr_std, snr_mean + snr_std,
    color='k', alpha=.2)
axes[1, 0].set(xlabel='Frequency [Hz]', ylabel='SNR',
               xlim=[fmin, fmax], ylim=[-1.5, 10])
cp.trim_axes(axes[1, 0])

# topography maps
mne.viz.plot_topomap(snrs_1f1_avg, epochs.info, vlim=(1, None),
                     axes=axes[0, 1])
axes[0, 1].set(title='1f1')

mne.viz.plot_topomap(snrs_1f2_avg, epochs.info, vlim=(1, None),
                     axes=axes[1, 1])
axes[1, 1].set(title='1f2')

plt.show()
# ----------------------------------------------------------------------------
