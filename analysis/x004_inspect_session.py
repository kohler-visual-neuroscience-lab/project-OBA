import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib import cleanplot as cp

"""
MoShams <MShamsCBR@gmail.com> Dec 28, 2022

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

# /// SET UP SOURCE DATA PATH AND PARAMETERS ///

eeg_file = '0005_20230113_105012.mff'
beh_file = '0005_20230113_104958.json'

# eeg_file = '0010_20230113_010843.mff'
# beh_file = '0010_20230113_130828.json'

# eeg_file = '0011_20230113_120420.mff'
# beh_file = '0011_20230113_120405.json'

# set the full path to the raw data
eeg_path = os.path.join('..', 'data', 'raw', eeg_file)
beh_path = os.path.join('..', 'data', 'raw', beh_file)
eeg = mne.io.read_raw_egi(eeg_path, preload=True)
beh_data = pd.read_json(beh_path)

# ----------------------------------------------------------------------------

# /// SET UP SAVE PATH AND PARAMETERS ///

save_path = os.path.join('..', 'result')
# extract subject's ID
sub_id = beh_file[:4]
# extract recoding date
rec_date = beh_file[5:13]
# ----------------------------------------------------------------------------

# /// READ AND CONFIGURE BEHAVIORAL DATA ///

# set number of blocks
n_blocks = 4
# extract number of trials
n_trials = beh_data.shape[0]
# convert tilt magnitudes to degrees
tilt_angle = (beh_data['tilt_magnitude'].values + 1) / 10
# ----------------------------------------------------------------------------

# /// READ AND CONFIGURE EEG DATA ///

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
# find the closest frequency bin to stimulation frequency
stim_freq1 = 7.5
i_bin_1f1 = np.argmin(abs(freqs - stim_freq1 * 1))
i_bin_2f1 = np.argmin(abs(freqs - stim_freq1 * 2))
i_bin_3f1 = np.argmin(abs(freqs - stim_freq1 * 3))
stim_freq2 = 12
i_bin_1f2 = np.argmin(abs(freqs - stim_freq2 * 1))
i_bin_2f2 = np.argmin(abs(freqs - stim_freq2 * 2))
i_bin_3f2 = np.argmin(abs(freqs - stim_freq2 * 3))
# index first harmonic of each stimulus frequency in snr array
# NOTE: the dimensions of snrs are: events/trials x channels x freq_bins
snrs_1f1 = snrs[:, :, i_bin_1f1]
snrs_1f2 = snrs[:, :, i_bin_1f2]
# average across events/trials
snrs_1f1_avg = snrs_1f1.mean(axis=0)
snrs_1f2_avg = snrs_1f2.mean(axis=0)

# index trials/events in a certain condition
i_cnd1 = events[:, 2] == 1
i_cnd2 = events[:, 2] == 2
i_cnd3 = events[:, 2] == 3
i_cnd4 = events[:, 2] == 4
# index trials/events in a certain condition at a certain frequency
snrs_cnd1_1f1 = snrs[i_cnd1, :, i_bin_1f1]
snrs_cnd2_1f1 = snrs[i_cnd2, :, i_bin_1f1]
snrs_cnd3_1f1 = snrs[i_cnd3, :, i_bin_1f1]
snrs_cnd4_1f1 = snrs[i_cnd4, :, i_bin_1f1]
snrs_cnd1_1f2 = snrs[i_cnd1, :, i_bin_1f2]
snrs_cnd2_1f2 = snrs[i_cnd2, :, i_bin_1f2]
snrs_cnd3_1f2 = snrs[i_cnd3, :, i_bin_1f2]
snrs_cnd4_1f2 = snrs[i_cnd4, :, i_bin_1f2]
# average across trials/events
snrs_cnd1_1f1_avg = snrs_cnd1_1f1.mean(axis=0)
snrs_cnd2_1f1_avg = snrs_cnd2_1f1.mean(axis=0)
snrs_cnd3_1f1_avg = snrs_cnd3_1f1.mean(axis=0)
snrs_cnd4_1f1_avg = snrs_cnd4_1f1.mean(axis=0)
snrs_cnd1_1f2_avg = snrs_cnd1_1f2.mean(axis=0)
snrs_cnd2_1f2_avg = snrs_cnd2_1f2.mean(axis=0)
snrs_cnd3_1f2_avg = snrs_cnd3_1f2.mean(axis=0)
snrs_cnd4_1f2_avg = snrs_cnd4_1f2.mean(axis=0)

# for each image at each frequency, subtract unattended map from attended to 
# obrtain "image boost" topography map
face_boost_1f1 = snrs_cnd1_1f1_avg - snrs_cnd3_1f1_avg
face_boost_1f2 = snrs_cnd2_1f2_avg - snrs_cnd4_1f2_avg
house_boost_1f1 = snrs_cnd4_1f1_avg - snrs_cnd2_1f1_avg
house_boost_1f2 = snrs_cnd3_1f2_avg - snrs_cnd1_1f2_avg

# ----------------------------------------------------------------------------

# +++ TEST +++

# make sure that number of events (in eeg file) matches the number of trials
# (in beh file)
assert events.shape[0] == beh_data.shape[0]
# ----------------------------------------------------------------------------

# @@@ PLOT BEHAVIORAL ANALYSES @@@

fig, axs = plt.subplots(4, 2, figsize=(10, 8), width_ratios=[3, 1])
fig.suptitle(f'Subject ID: {sub_id} – Behavioral performance')
cp.prep4ai()

# set up plot parameters
mrksize = 4  # marker size
trials = np.arange(n_trials) + 1  # x values

# events as a function of trials
x_evnt1 = np.nonzero(events[:, 2] == 1)[0] + 1
axs[0, 0].plot(x_evnt1, 1 * np.ones(len(x_evnt1)), 'o', markersize=mrksize,
               markerfacecolor='k', markeredgecolor='none')
x_evnt2 = np.nonzero(events[:, 2] == 2)[0] + 1
axs[0, 0].plot(x_evnt2, 2 * np.ones(len(x_evnt2)), 'o', markersize=mrksize,
               markerfacecolor='k', markeredgecolor='none')
x_evnt3 = np.nonzero(events[:, 2] == 3)[0] + 1
axs[0, 0].plot(x_evnt3, 3 * np.ones(len(x_evnt3)), 'o', markersize=mrksize,
               markerfacecolor='k', markeredgecolor='none')
x_evnt4 = np.nonzero(events[:, 2] == 4)[0] + 1
axs[0, 0].plot(x_evnt4, 4 * np.ones(len(x_evnt4)), 'o', markersize=mrksize,
               markerfacecolor='k', markeredgecolor='none')
axs[0, 0].set(yticks=[1, 2, 3, 4],
              yticklabels=['CND1', 'CND2', 'CND3', 'CND4'],
              ylabel='Events', xlim=[0 - 2, n_trials + 1], ylim=[1 - .5, 4.5])
cp.add_shades(axs[0, 0], n_blocks, n_trials)
axs[0, 0].text(0 * 32 + 3, 4.6, 'Block1')
axs[0, 0].text(1 * 32 + 3, 4.6, 'Block2')
axs[0, 0].text(2 * 32 + 3, 4.6, 'Block3')
axs[0, 0].text(3 * 32 + 3, 4.6, 'Block4')
cp.trim_axes(axs[0, 0])

# performances over session
line_cumperf, = axs[1, 0].plot(trials, beh_data['cummulative_performance'],
                               color='silver')
line_runperf, = axs[1, 0].plot(trials, beh_data['running_performance'],
                               color='k')
axs[1, 0].set(ylabel='Performance [%]',
              xlim=[0 - 2, n_trials + 1], ylim=[0 - 5, 100])
leg = axs[1, 0].legend([line_cumperf, line_runperf], ['cum perf', 'run perf'])
leg.get_frame().set_linewidth(0)
cp.add_shades(axs[1, 0], n_blocks, n_trials)
cp.trim_axes(axs[1, 0])

# RT over session
axs[2, 0].plot(trials, beh_data['avg_rt'], 'o', markerfacecolor='k',
               markeredgecolor='none', markersize=mrksize)
axs[2, 0].set_yticks(range(0, 1000 + 1, 250))
axs[2, 0].set(ylabel='RT [ms]',
              xlim=[0 - 2, n_trials + 1], ylim=[0 - 50, 1000])
cp.add_shades(axs[2, 0], n_blocks, n_trials)
cp.trim_axes(axs[2, 0])

# tilt angles over session
axs[3, 0].plot(trials, tilt_angle, color='k')
axs[3, 0].set(xlabel='Trials', ylabel='Tilt angle [deg]',
              xlim=[0 - 2, n_trials + 1], ylim=[0 - .3, 6])
cp.add_shades(axs[3, 0], n_blocks, n_trials)
cp.trim_axes(axs[3, 0])

# leave this subplot empty
axs[0, 1].axis('off')

# leave this subplot empty
axs[1, 1].axis('off')

# RT histogram
hist_bins = range(0, 1000, 100)
axs[2, 1].hist(beh_data['avg_rt'], facecolor='k', bins=hist_bins)
axs[2, 1].set_xticks(range(0, 1000 + 1, 250))
axs[2, 1].set(xticks=range(0, 1000 + 1, 250),
              xlabel='RT [ms]', ylabel='Count',
              xlim=[0 - 30, 1000], ylim=[0 - .5, 30])
cp.trim_axes(axs[2, 1])

# leave this subplot empty
axs[3, 1].axis('off')

# save figure
plt.savefig(os.path.join(save_path, f'{sub_id}_{rec_date}_behavior.pdf'))
# ----------------------------------------------------------------------------

# @@@ PLOT PSD, SNR, 1F1, 1F2 @@@

fig, axes = plt.subplots(2, 2, figsize=(10, 6), width_ratios=[3, 1])
fig.suptitle(f'Subject ID: {sub_id} – PSD, SNR, and Laterality')
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
vmin = 1
vmax = 5
im, _ = mne.viz.plot_topomap(snrs_1f1_avg, epochs.info, vlim=(vmin, vmax),
                             axes=axes[0, 1], show=False)
axes[0, 1].set(title='1f1')
cp.add_snr_colorbar(fig, axes[0, 1], im)

im, _ = mne.viz.plot_topomap(snrs_1f2_avg, epochs.info, vlim=(vmin, vmax),
                             axes=axes[1, 1], show=False)
axes[1, 1].set(title='1f2')
cp.add_snr_colorbar(fig, axes[1, 1], im)

# save figure
plt.savefig(os.path.join(save_path, f'{sub_id}_'
                                    f'{rec_date}_psd_snr_laterality.pdf'))
# ----------------------------------------------------------------------------

# @@@ PLOT TOPOGRAPHY MAPS FOR EACH CONDITION AT EACH FREQUENCY @@@

# set up the min and max of the color map (insert 'None' to pass)
vmin = 1
vmax = 10

fig, axes = plt.subplots(4, 3, figsize=(7.5, 11.5))
fig.suptitle(f'Subject ID: {sub_id} – Topography map of each condition at '
             f'each frequency')
cp.prep4ai()

# @@@ Face boost pairs
# Topography map of CND1, FH(F), 1f1:
im, _ = mne.viz.plot_topomap(snrs_cnd1_1f1_avg, epochs.info, axes=axes[0, 0],
                             vlim=(vmin, vmax), show=False)
axes[0, 0].set(title='CND1 FH(F) 1f1')
cp.add_snr_colorbar(fig, axes[0, 0], im)
# Topography map of CND3, FH(H), 1f1:
im, _ = mne.viz.plot_topomap(snrs_cnd3_1f1_avg, epochs.info, axes=axes[0, 1],
                             vlim=(vmin, vmax), show=False)
axes[0, 1].set(title='CND3 FH(H) 1f1')
cp.add_snr_colorbar(fig, axes[0, 1], im)
# Topography map of CND2, HF(F), 1f2:
im, _ = mne.viz.plot_topomap(snrs_cnd2_1f2_avg, epochs.info, axes=axes[1, 0],
                             vlim=(vmin, vmax), show=False)
axes[1, 0].set(title='CND2 HF(F) 1f2')
cp.add_snr_colorbar(fig, axes[1, 0], im)
# Topography map of CND4, HF(H), 1f2:
im, _ = mne.viz.plot_topomap(snrs_cnd4_1f2_avg, epochs.info, axes=axes[1, 1],
                             vlim=(vmin, vmax), show=False)
axes[1, 1].set(title='CND4 HF(H) 1f2')
cp.add_snr_colorbar(fig, axes[1, 1], im)

# @@@ House boost pairs
# Topography map of CND4, HF(H), 1f1:
im, _ = mne.viz.plot_topomap(snrs_cnd4_1f1_avg, epochs.info, axes=axes[2, 0],
                             vlim=(vmin, vmax), show=False)
axes[2, 0].set(title='CND4 HF(H) 1f1')
cp.add_snr_colorbar(fig, axes[2, 0], im)
# Topography map of CND2, HF(F), 1f1:
im, _ = mne.viz.plot_topomap(snrs_cnd2_1f1_avg, epochs.info, axes=axes[2, 1],
                             vlim=(vmin, vmax), show=False)
axes[2, 1].set(title='CND2 HF(F) 1f1')
cp.add_snr_colorbar(fig, axes[2, 1], im)
# Topography map of CND3, FH(H), 1f2:
im, _ = mne.viz.plot_topomap(snrs_cnd3_1f2_avg, epochs.info, axes=axes[3, 0],
                             vlim=(vmin, vmax), show=False)
axes[3, 0].set(title='CND3 FH(H) 1f2')
cp.add_snr_colorbar(fig, axes[3, 0], im)
# Topography map of CND1, FH(F), 1f2:
im, _ = mne.viz.plot_topomap(snrs_cnd1_1f2_avg, epochs.info, axes=axes[3, 1],
                             vlim=(vmin, vmax), show=False)
axes[3, 1].set(title='CND1 FH(F) 1f2')
cp.add_snr_colorbar(fig, axes[3, 1], im)

# @@@ ADD OBJECT BOOST MAPS
# set up the min and max of the color map (insert 'None' to pass)
vmin = -3
vmax = 3
# Topography map of face boost at 1f1:
im, _ = mne.viz.plot_topomap(face_boost_1f1, epochs.info, axes=axes[0, 2],
                             vlim=(vmin, vmax), show=False)
axes[0, 2].set(title='Face boost 1f1')
cp.add_snr_colorbar(fig, axes[0, 2], im)
# Topography map of face boost at 1f2:
im, _ = mne.viz.plot_topomap(face_boost_1f2, epochs.info, axes=axes[1, 2],
                             vlim=(vmin, vmax), show=False)
axes[1, 2].set(title='Face boost 1f2')
cp.add_snr_colorbar(fig, axes[1, 2], im)

# Topography map of house boost at 1f1:
im, _ = mne.viz.plot_topomap(house_boost_1f1, epochs.info, axes=axes[2, 2],
                             vlim=(vmin, vmax), show=False)
axes[2, 2].set(title='House boost 1f1')
cp.add_snr_colorbar(fig, axes[2, 2], im)
# Topography map of house boost at 1f2:
im, _ = mne.viz.plot_topomap(house_boost_1f2, epochs.info, axes=axes[3, 2],
                             vlim=(vmin, vmax), show=False)
axes[3, 2].set(title='House boost 1f2')
cp.add_snr_colorbar(fig, axes[3, 2], im)

# save figure
plt.savefig(os.path.join(save_path, f'{sub_id}_'
                                    f'{rec_date}_topomap_all_conditions.pdf'))
# ----------------------------------------------------------------------------
