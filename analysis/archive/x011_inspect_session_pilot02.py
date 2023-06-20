import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib import cleanplot as cp

"""
MoShams <MShamsCBR@gmail.com> May 13, 2023

- Inspects the recorded behavioral and EEG data of a single session.
- Generates two figures: behavioral analysis, SSVEP analysis

"""


# /// functions
def snr_spectrum(psd, noise_n_neighbor_freqs=3, noise_skip_neighbor_freqs=1):
    # Construct a kernel that calculates the mean of the neighboring
    # frequencies (noisy neighbors!)
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

# (N=4)

# eeg_file = '0004_20230519_010426_pilot02.mff'
# beh_file = '0004_20230519_120423_pilot02.json'

# eeg_file = '0012_20230519_114513_pilot02.mff'
# beh_file = '0012_20230519_104510_pilot02.json'

# eeg_file = '0013_20230519_122443_pilot02.mff'
# beh_file = '0013_20230519_112440_pilot02.json'

eeg_file = '1191_20230519_013653_pilot02.mff'
beh_file = '1191_20230519_123650_pilot02.json'


# set the full path to the raw data
eeg_path = os.path.join('..', 'data', 'pilot02', 'raw', eeg_file)
beh_path = os.path.join('..', 'data', 'pilot02', 'raw', beh_file)
eeg = mne.io.read_raw_egi(eeg_path, preload=True)
beh_data = pd.read_json(beh_path)

# ----------------------------------------------------------------------------

# /// SET UP SAVE PATH AND PARAMETERS ///

save_path = os.path.join('..', 'result', 'pilot02')
# extract subject's ID
sub_id = beh_file[:4]
# extract recoding date
rec_date = beh_file[5:13]
# ----------------------------------------------------------------------------

# /// READ AND CONFIGURE BEHAVIORAL DATA ///

# extract number of trials
n_trials = beh_data.shape[0]

# set number of blocks
# n_blocks = 4
n_blocks = n_trials / 32  # from session 4 onward, we recorded 2 blocks of
# 32 trials

# convert tilt magnitudes to degrees
tilt_angle = (beh_data['tilt_magnitude'].values + 1) / 10
# ----------------------------------------------------------------------------

# /// READ AND CONFIGURE EEG DATA ///

eeg.info['line_freq'] = 60.
# set montage
montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
eeg.set_montage(montage, match_alias=True)
easycap_montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
# easycap_montage.plot()
# set common average reference
eeg.set_eeg_reference('average', projection=False, verbose=False)
# apply bandpass filter
eeg.filter(l_freq=0.1, h_freq=None, fir_design='firwin', verbose=False)

# /// extract events
events = mne.find_events(eeg, stim_channel=['CND1', 'CND2'])
event_id = eeg.event_id

# /// reshape data into epochs
tmin = 1  # in sec wrt event times
tmax = 7  # in sec wrt event times
epochs = mne.Epochs(eeg, events=events,
                    event_id=[event_id['CND1'], event_id['CND2']],
                    tmin=tmin, tmax=tmax, baseline=None, verbose=False)

# /// calculate signal-to-noise ratio (SNR) and power spectral density (PSD)
tmin = tmin
tmax = tmax
fmin = 1.
fmax = 20.
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
i_cnd1 = beh_data['condition_num'] == 1
i_cnd2 = beh_data['condition_num'] == 2

# index trials/events in a certain condition at a certain frequency
snrs_cnd1_1f1 = snrs[i_cnd1, :, i_bin_1f1]
snrs_cnd2_1f1 = snrs[i_cnd2, :, i_bin_1f1]
snrs_cnd1_1f2 = snrs[i_cnd1, :, i_bin_1f2]
snrs_cnd2_1f2 = snrs[i_cnd2, :, i_bin_1f2]

# average across trials/events
snrs_cnd1_1f1_avg = snrs_cnd1_1f1.mean(axis=0)
snrs_cnd2_1f1_avg = snrs_cnd2_1f1.mean(axis=0)
snrs_cnd1_1f2_avg = snrs_cnd1_1f2.mean(axis=0)
snrs_cnd2_1f2_avg = snrs_cnd2_1f2.mean(axis=0)

# for each image at each frequency, subtract unattended map from attended to
# obrtain "image boost" topography map
face_boost_1f1 = snrs_cnd1_1f1_avg - snrs_cnd2_1f1_avg
house_boost_1f2 = snrs_cnd2_1f2_avg - snrs_cnd1_1f2_avg

# average across all channels
face_boost = np.mean(face_boost_1f1)
house_boost = np.mean(house_boost_1f2)

# average across occipital channels
occ_ch = ['E66', 'E69', 'E70', 'E71', 'E73', 'E74', 'E75',
          'E76', 'E81', 'E82', 'E83', 'E84', 'E88', 'E89']
ind_occ = np.nonzero(np.isin(epochs.info.ch_names, occ_ch))
face_boost_occ = np.mean(face_boost_1f1[ind_occ])
house_boost_occ = np.mean(house_boost_1f2[ind_occ])
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
axs[0, 0].set(yticks=[1, 2],
              yticklabels=['CND1', 'CND2'],
              ylabel='Events', xlim=[0 - 2, n_trials + 1],
              ylim=[1 - .25, 2 + .25])
cp.add_shades(axs[0, 0], n_blocks, n_trials)
axs[0, 0].text(0 * 32 + 3, 4.6, 'Block1')
axs[0, 0].text(1 * 32 + 3, 4.6, 'Block2')
axs[0, 0].get_xaxis().set_visible(False)
axs[0, 0].spines['bottom'].set_visible(False)
cp.trim_axes(axs[0, 0])

# performances over session
line_cumperf, = axs[1, 0].plot(trials, beh_data['cummulative_performance'],
                               color='silver')
line_runperf, = axs[1, 0].plot(trials, beh_data['running_performance'],
                               color='k')
axs[1, 0].set(ylabel='Performance [%]',
              xlim=[0 - 2, n_trials + 1], ylim=[0, 100])
leg = axs[1, 0].legend([line_cumperf, line_runperf],
                       ['Cummulative perf.', 'Running perf.'])
leg.get_frame().set_linewidth(0)
axs[1, 0].get_xaxis().set_visible(False)
axs[1, 0].spines['bottom'].set_visible(False)
cp.add_shades(axs[1, 0], n_blocks, n_trials)
cp.trim_axes(axs[1, 0])

# RT over session
axs[2, 0].plot(trials, beh_data['avg_rt'], 'o', markerfacecolor='k',
               markeredgecolor='none', markersize=mrksize)
axs[2, 0].set_yticks(range(0, 1000 + 1, 250))
axs[2, 0].set(ylabel='RT [ms]',
              xlim=[0 - 2, n_trials + 1], ylim=[0, 1000 + 25])
cp.add_shades(axs[2, 0], n_blocks, n_trials)
axs[2, 0].get_xaxis().set_visible(False)
axs[2, 0].spines['bottom'].set_visible(False)
cp.trim_axes(axs[2, 0])

# tilt angles over session
axs[3, 0].plot(trials, tilt_angle, color='k')
axs[3, 0].set(xlabel='Trials', ylabel='Tilt angle [deg]',
              xticks=[0, 32, 64],
              xlim=[0, n_trials], ylim=[0, 6])
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
              xlim=[0, 1000], ylim=[0, 30])
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

# -------------------------
# PSD spectrum

axes[0, 0].plot(freqs[freq_range], psds_mean, color='k')
axes[0, 0].fill_between(
    freqs[freq_range], psds_mean - psds_std, psds_mean + psds_std,
    color='k', alpha=.2)
axes[0, 0].set(ylabel='PSD [dB]', xlim=[fmin, fmax])
cp.trim_axes(axes[0, 0])

# -------------------------
# SNR spectrum

axes[1, 0].plot(freqs[freq_range], snr_mean, color='k')
axes[1, 0].fill_between(
    freqs[freq_range], snr_mean - snr_std, snr_mean + snr_std,
    color='k', alpha=.2)
axes[1, 0].set(xlabel='Frequency [Hz]', ylabel='SNR',
               xlim=[fmin, fmax])
cp.trim_axes(axes[1, 0])

# -------------------------
# topography maps

vmin = 1
vmax = np.max([snrs_1f1_avg, snrs_1f2_avg])

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
fig, axes = plt.subplots(2, 3, figsize=(7.5, 6))
fig.suptitle(f'Subject ID: {sub_id} – Topography map of each condition at '
             f'each frequency')
cp.prep4ai()

# -------------------------
# @@@ face boost pairs

# Topography map of CND1 1f1:
im, _ = mne.viz.plot_topomap(snrs_cnd1_1f1_avg, epochs.info, axes=axes[0, 0],
                             vlim=(vmin, vmax), show=False)
axes[0, 0].set(title='CND1 1f1')
cp.add_snr_colorbar(fig, axes[0, 0], im)
# Topography map of CND1 1f2:
im, _ = mne.viz.plot_topomap(snrs_cnd2_1f1_avg, epochs.info, axes=axes[0, 1],
                             vlim=(vmin, vmax), show=False)
axes[0, 1].set(title='CND2 1f1')
cp.add_snr_colorbar(fig, axes[0, 1], im)

# -------------------------
# @@@ house boost pairs

# Topography map of CND2 1f2:
im, _ = mne.viz.plot_topomap(snrs_cnd2_1f2_avg, epochs.info, axes=axes[1, 0],
                             vlim=(vmin, vmax), show=False)
axes[1, 0].set(title='CND2 1f2')
cp.add_snr_colorbar(fig, axes[1, 0], im)
# Topography map of CND2, 1f1:
im, _ = mne.viz.plot_topomap(snrs_cnd1_1f2_avg, epochs.info, axes=axes[1, 1],
                             vlim=(vmin, vmax), show=False)
axes[1, 1].set(title='CND1 1f2')
cp.add_snr_colorbar(fig, axes[1, 1], im)

# -------------------------
# @@@ ADD OBJECT BOOST MAPS

# set up the min and max of the color map (insert 'None' to pass)
vmin_diff = -vmax/2
vmax_diff = vmax/2
# Topography map of red135 boost at 1f1:
im, _ = mne.viz.plot_topomap(face_boost_1f1, epochs.info, axes=axes[0, 2],
                             vlim=(vmin_diff, vmax_diff), show=False)
axes[0, 2].set(title='face boost 1f1')
cp.add_snr_colorbar(fig, axes[0, 2], im)
# Topography map of red135 boost at 1f2:
im, _ = mne.viz.plot_topomap(house_boost_1f2, epochs.info, axes=axes[1, 2],
                             vlim=(vmin_diff, vmax_diff), show=False)
axes[1, 2].set(title='house boost 1f2')
cp.add_snr_colorbar(fig, axes[1, 2], im)

# save figure
plt.savefig(os.path.join(save_path, f'{sub_id}_'
                                    f'{rec_date}_topomap_all_conditions.pdf'))
# ----------------------------------------------------------------------------

# # @@@ PLOT AVERAGE SNR CHANNELS @@@
#
# fig, axs = plt.subplots(1, 2, figsize=(5, 4), sharey=True)
# fig.suptitle(f'Subject ID: {sub_id} – Avg. SNR improvement')
# axs[0].bar([1, 2], [red135_boost, blue45_boost], color='grey')
# axs[0].set(title='All channels', xticks=[1, 2],
#            xticklabels=['red135', 'blue45'], ylabel='SNR improvement')
# cp.trim_axes(axs[0])
# cp.prep4ai()
#
# axs[1].bar([1, 2], [red135_boost_occ, blue45_boost_occ], color='grey')
# axs[1].set(title='Occipital channels', xticks=[1, 2],
#            xticklabels=['red135', 'blue45'])
# cp.trim_axes(axs[1])
# cp.prep4ai()
#
# plt.tight_layout()
#
# # save figure
# plt.savefig(os.path.join(save_path, f'{sub_id}_'
#                                     f'{rec_date}_avg_SNR_improvement.pdf'))

# ----------------------------------------------------------------------------
#
# # @@@ PLOT BOOST ACROSS TRIALS @@@
# ind_occ = [65, 68, 69, 70, 72, 73, 74, 75, 80, 81, 82, 83, 87, 88]
# # the aim is to average over 8 trials
# # how many 'blocks' of 8 trials will there be?
# kk = int(n_trials / 4 / 8)
#
# ch_avg_cnd1 = snrs_cnd1_1f1[:, ind_occ].mean(axis=1)
# ch_avg_cnd2 = snrs_cnd2_1f1[:, ind_occ].mean(axis=1)
# ch_avg_cnd3 = snrs_cnd3_1f1[:, ind_occ].mean(axis=1)
# ch_avg_cnd4 = snrs_cnd4_1f1[:, ind_occ].mean(axis=1)
# ch_avg_cnd1_blocked_1f1 = np.mean(np.reshape(ch_avg_cnd1, (kk, 8)), axis=1)
# ch_avg_cnd2_blocked_1f1 = np.mean(np.reshape(ch_avg_cnd2, (kk, 8)), axis=1)
# ch_avg_cnd3_blocked_1f1 = np.mean(np.reshape(ch_avg_cnd3, (kk, 8)), axis=1)
# ch_avg_cnd4_blocked_1f1 = np.mean(np.reshape(ch_avg_cnd4, (kk, 8)), axis=1)
#
# ch_avg_cnd2 = snrs_cnd2_1f2[:, ind_occ].mean(axis=1)
# ch_avg_cnd3 = snrs_cnd3_1f2[:, ind_occ].mean(axis=1)
# ch_avg_cnd1 = snrs_cnd1_1f2[:, ind_occ].mean(axis=1)
# ch_avg_cnd4 = snrs_cnd4_1f2[:, ind_occ].mean(axis=1)
# ch_avg_cnd1_blocked_1f2 = np.mean(np.reshape(ch_avg_cnd1, (kk, 8)), axis=1)
# ch_avg_cnd2_blocked_1f2 = np.mean(np.reshape(ch_avg_cnd2, (kk, 8)), axis=1)
# ch_avg_cnd3_blocked_1f2 = np.mean(np.reshape(ch_avg_cnd3, (kk, 8)), axis=1)
# ch_avg_cnd4_blocked_1f2 = np.mean(np.reshape(ch_avg_cnd4, (kk, 8)), axis=1)
#
# red_boost_1f1_blocked = ch_avg_cnd1_blocked_1f1 - ch_avg_cnd3_blocked_1f1
# red_boost_1f2_blocked = ch_avg_cnd2_blocked_1f2 - ch_avg_cnd4_blocked_1f2
#
# blue_boost_1f1_blocked = ch_avg_cnd4_blocked_1f1 - ch_avg_cnd2_blocked_1f1
# blue_boost_1f2_blocked = ch_avg_cnd3_blocked_1f2 - ch_avg_cnd1_blocked_1f2
#
# red_boost_occ_blocked = (red_boost_1f1_blocked + red_boost_1f2_blocked)/2
# blue_boost_occ_blocked = (blue_boost_1f1_blocked + blue_boost_1f2_blocked)/2
#
# fig, ax = plt.subplots()
# ax.plot(red_boost_occ_blocked)
# ax.plot(blue_boost_occ_blocked)
