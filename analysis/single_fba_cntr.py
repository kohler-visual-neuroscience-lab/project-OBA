"""
Mohammad Shams <MoShamsCBR@gmail.com>
June 19, 2023

Inspect a single session recorded from URPP students over May and June
2023 as oba-ssvep-cycle02. We ran four tasks in random order:
FBA-central, FBA-peripheral, OBA-central, OBA-peripheral.
Central and Peripheral refer to the position of the flickering images.

"""

import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib import cleanplot as cp


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
# /// SET-UP DATA AND RESULT PATHS ///

subject_id = 5007
task_name = 'fba_per'
data_folder = os.path.join('..', 'data', 'cycle02')
result_folder = os.path.join('..', 'result', 'cycle02')

all_files = os.listdir(data_folder)
beh_file = [file for file in all_files if file.endswith('.json') and
            f'{task_name}' in file and
            f'{subject_id}' in file][0]
eeg_file = [file for file in all_files if file.endswith('.mff') and
            f'{task_name}' in file and
            f'{subject_id}' in file][0]

# set the full path to the raw data
eeg_path = os.path.join(data_folder, eeg_file)
beh_path = os.path.join(data_folder, beh_file)
eeg = mne.io.read_raw_egi(eeg_path, preload=True)
beh_data = pd.read_json(beh_path)

# extract subject's ID
sub_id = beh_file[:4]
# extract recoding date
rec_date = beh_file[5:13]
# ----------------------------------------------------------------------------

# /// READ AND CONFIGURE BEHAVIORAL DATA ///

# extract number of trials
n_trials = beh_data.shape[0]

# extract number of blocks
n_blocks = beh_data['block_num'].max()
n_trials_per_block = int(n_trials/n_blocks)

# convert tilt magnitudes to degrees
tilt_angle = (beh_data['tilt_magnitude'].values + 1) / 10

# extract image-freq links
# The first frequency in the 1 x 2 array in the 'Frequency column' is for Blue
# and the second one for Red.
f_label = []
if beh_data['Frequency_tags'][0][0] == 7.5:
    if 'fba' in task_name:
        f_label = ['Blue', 'Red']
    elif 'oba' in task_name:
        f_label = ['Face', 'House']
elif beh_data['Frequency_tags'][0][0] == 12:
    if 'fba' in task_name:
        f_label = ['Red', 'Blue']
    elif 'oba' in task_name:
        f_label = ['House', 'Face']

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
fmax = 50.
sampling_freq = epochs.info['sfreq']

# calculate PSD
spectrum = epochs.compute_psd('welch',
                              n_fft=int(sampling_freq * (tmax - tmin)),
                              n_overlap=0, n_per_seg=None, tmin=tmin,
                              tmax=tmax, fmin=fmin, fmax=fmax, window='boxcar',
                              verbose=False)
psds, freqs = spectrum.get_data(return_freqs=True)

# calculate SNR
# snrs: [64 trials] x [129 channels] x [115 freq. bins]
snrs = snr_spectrum(psds, noise_n_neighbor_freqs=3,
                    noise_skip_neighbor_freqs=1)

# define a freq range
# freq_range: [103 freq. bins]
freq_range = range(np.where(np.floor(freqs) == 1.)[0][0],
                   np.where(np.ceil(freqs) == fmax - 1)[0][0])

# average PSD across all trials and all channels within the desired freq range
# psds_mean/std: [103 freq. bins]
psds_plot = 10 * np.log10(psds)
psds_mean = psds_plot.mean(axis=(0, 1))[freq_range]
psds_std = psds_plot.std(axis=(0, 1))[freq_range]

# average SNR across all trials and all channels within the desired freq range
# snr_mean/std: [103 freq. bins]
snr_mean = snrs.mean(axis=(0, 1))[freq_range]
snr_std = snrs.std(axis=(0, 1))[freq_range]

# /// prepare data for topography maps
# find the closest frequency bin to stimulation frequency
stim_freq1 = 7.5
i_bin_1f1 = np.argmin(abs(freqs - stim_freq1 * 1))
stim_freq2 = 12
i_bin_1f2 = np.argmin(abs(freqs - stim_freq2 * 1))
# index first harmonic of each stimulus frequency in snr array and average
# across events/trials
# topo_1fX: [129 channels]
topo_1f1 = snrs.mean(axis=0)[:, i_bin_1f1]
topo_1f2 = snrs.mean(axis=0)[:, i_bin_1f2]

# index trials/events in a certain condition
i_cnd1 = beh_data['condition_num'] == 1
i_cnd2 = beh_data['condition_num'] == 2

# index trials/events in a certain condition at a certain frequency
# topo_cndX_1fX:  [129 channels]
topo_cnd1_1f1 = snrs[i_cnd1, :, i_bin_1f1].mean(axis=0)
topo_cnd2_1f1 = snrs[i_cnd2, :, i_bin_1f1].mean(axis=0)
topo_cnd1_1f2 = snrs[i_cnd1, :, i_bin_1f2].mean(axis=0)
topo_cnd2_1f2 = snrs[i_cnd2, :, i_bin_1f2].mean(axis=0)

# for each image at each frequency, subtract unattended map from attended
# map to obrtain "freq boost" topography map
topo_boost_1f1 = topo_cnd1_1f1 - topo_cnd2_1f1
topo_boost_1f2 = topo_cnd2_1f2 - topo_cnd1_1f2

# average across all channels
boost_1f1_allCh = topo_boost_1f1.mean()
boost_1f2_allCh = topo_boost_1f2.mean()

# average across occipital channels
occCh = ['E66', 'E69', 'E70', 'E71', 'E73', 'E74', 'E75',
         'E76', 'E81', 'E82', 'E83', 'E84', 'E88', 'E89']
ind_occCh = np.nonzero(np.isin(epochs.info.ch_names, occCh))[0]
boost_1f1_occCh = topo_boost_1f1[ind_occCh].mean()
boost_1f2_occCh = topo_boost_1f2[ind_occCh].mean()

# extract the strongest channel
ind_bestCh_1f1 = np.argmax(topo_1f1)
ind_bestCh_1f2 = np.argmax(topo_1f2)
boost_1f1_bestCh = topo_boost_1f1[ind_bestCh_1f1]
boost_1f2_bestCh = topo_boost_1f2[ind_bestCh_1f2]

# average SNR for each condition acroos occipital channels
snr_cnd1 = snrs[np.ix_(i_cnd1, ind_occCh, freq_range)].mean(axis=(0, 1))
snr_cnd2 = snrs[np.ix_(i_cnd2, ind_occCh, freq_range)].mean(axis=(0, 1))
# ----------------------------------------------------------------------------

# +++ TEST +++

# make sure that number of events (in eeg file) matches the number of trials
# (in beh file)
assert events.shape[0] == beh_data.shape[0]
# ----------------------------------------------------------------------------

# @@@ PLOT BEHAVIORAL ANALYSES @@@

fig, axs = plt.subplots(4, 2, figsize=(10, 8), width_ratios=[3, 1])
fig.suptitle(f'Behavioral performance (Subject:{sub_id})')
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
axs[0, 0].text(0 * n_trials_per_block + 3, 4.6, 'Block1')
axs[0, 0].text(1 * n_trials_per_block + 3, 4.6, 'Block2')
axs[0, 0].get_xaxis().set_visible(False)
axs[0, 0].spines['bottom'].set_visible(False)
cp.trim_axes(axs[0, 0])

# performances over session
line_cumperf, = axs[1, 0].plot(trials, beh_data['cummulative_performance'],
                               color='silver')
line_runperf, = axs[1, 0].plot(trials, beh_data['running_performance'],
                               color='k')
axs[1, 0].set(ylabel='Performance [%]',
              xlim=[0 - 2, n_trials + 1], ylim=[0, 105])
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
              xticks=range(0, n_trials + 1, n_trials_per_block),
              xlim=[0, n_trials])
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
plt.savefig(os.path.join(result_folder,
                         f'{sub_id}_{task_name}_Fig01_behavior.pdf'))
# ----------------------------------------------------------------------------

# @@@ PLOT PSD, SNR, 1F1, 1F2 @@@

fig, axes = plt.subplots(2, 2, figsize=(10, 6), width_ratios=[3, 1])
fig.suptitle(f'SNR and Topo-map (Subject:{sub_id})')
cp.prep4ai()

# -------------------------
# # PSD spectrum
#
# axes[0, 0].plot(freqs[freq_range], psds_mean, color='k')
# axes[0, 0].fill_between(
#     freqs[freq_range], psds_mean - psds_std, psds_mean + psds_std,
#     color='k', alpha=.2)
# axes[0, 0].set(ylabel='PSD [dB]', xlim=[fmin, fmax])
# cp.trim_axes(axes[0, 0])

# -------------------------
# SNR spectrum (avg. across all channels)

fmin_show = 5
fmax_show = 15
fstep = 1

axes[0, 0].plot(freqs[freq_range], snr_mean, color='k')
axes[0, 0].fill_between(
    freqs[freq_range], snr_mean - snr_std, snr_mean + snr_std,
    color='k', alpha=.2)
axes[0, 0].set(ylabel='SNR',
               xticks=np.arange(fmin_show, fmax_show + fstep, fstep),
               xlim=[fmin_show, fmax_show])
cp.trim_axes(axes[0, 0])

# -------------------------
# SNR spectrum (avg. across occipital channels)

axes[1, 0].plot(freqs[freq_range], snr_cnd1, color='darkcyan',
                label='cnd1-occCh')
axes[1, 0].plot(freqs[freq_range], snr_cnd2, color='darkorchid',
                label='cnd2-occCh')
axes[1, 0].set(xlabel='Frequency [Hz]', ylabel='SNR',
               xticks=np.arange(fmin_show, fmax_show + fstep, fstep),
               xlim=[fmin_show, fmax_show])
leg = axes[1, 0].legend()
leg.get_frame().set_linewidth(0)
cp.trim_axes(axes[1, 0])

# -------------------------
# topography maps

vmin = 1
vmax = np.max([topo_1f1, topo_1f2])

im, _ = mne.viz.plot_topomap(topo_1f1, epochs.info, vlim=(vmin, vmax),
                             axes=axes[0, 1], show=False)
axes[0, 1].set(title='f1')
cp.add_snr_colorbar(fig, axes[0, 1], im)

im, _ = mne.viz.plot_topomap(topo_1f2, epochs.info, vlim=(vmin, vmax),
                             axes=axes[1, 1], show=False)
axes[1, 1].set(title='f2')
cp.add_snr_colorbar(fig, axes[1, 1], im)

# save figure
plt.savefig(os.path.join(result_folder,
                         f'{sub_id}_{task_name}_Fig02_SNR_Topomap.pdf'))
# ----------------------------------------------------------------------------

# @@@ PLOT TOPOGRAPHY MAPS FOR EACH CONDITION AT EACH FREQUENCY @@@

# set up the min and max of the color map (insert 'None' to pass)
fig, axes = plt.subplots(2, 3, figsize=(7.5, 6))
fig.suptitle(f'Topography map at each condition and '
             f'each frequency (Subject:{sub_id})')
cp.prep4ai()

# -------------------------
# @@@ face boost pairs

# Topography map of CND1 1f1:
im, _ = mne.viz.plot_topomap(topo_cnd1_1f1, epochs.info, axes=axes[0, 0],
                             vlim=(vmin, vmax), show=False)
axes[0, 0].set(title='CND1 f1')
cp.add_snr_colorbar(fig, axes[0, 0], im)
# Topography map of CND1 1f2:
im, _ = mne.viz.plot_topomap(topo_cnd2_1f1, epochs.info, axes=axes[0, 1],
                             vlim=(vmin, vmax), show=False)
axes[0, 1].set(title='CND2 f1')
cp.add_snr_colorbar(fig, axes[0, 1], im)

# -------------------------
# @@@ house boost pairs

# Topography map of CND2 1f2:
im, _ = mne.viz.plot_topomap(topo_cnd2_1f2, epochs.info, axes=axes[1, 0],
                             vlim=(vmin, vmax), show=False)
axes[1, 0].set(title='CND2 f2')
cp.add_snr_colorbar(fig, axes[1, 0], im)
# Topography map of CND2, 1f1:
im, _ = mne.viz.plot_topomap(topo_cnd1_1f2, epochs.info, axes=axes[1, 1],
                             vlim=(vmin, vmax), show=False)
axes[1, 1].set(title='CND1 f2')
cp.add_snr_colorbar(fig, axes[1, 1], im)

# -------------------------
# @@@ ADD OBJECT BOOST MAPS

# set up the min and max of the color map (insert 'None' to pass)
vmin_diff = -vmax
vmax_diff = vmax
# Topography map of 1f1 boost:
im, _ = mne.viz.plot_topomap(topo_boost_1f1, epochs.info, axes=axes[0, 2],
                             vlim=(vmin_diff, vmax_diff), show=False)
axes[0, 2].set(title=f'f1 boost ({f_label[0]})')
cp.add_snr_colorbar(fig, axes[0, 2], im)
# Topography map of 1f2 boost:
im, _ = mne.viz.plot_topomap(topo_boost_1f2, epochs.info, axes=axes[1, 2],
                             vlim=(vmin_diff, vmax_diff), show=False)
axes[1, 2].set(title=f'f2 boost ({f_label[1]})')
cp.add_snr_colorbar(fig, axes[1, 2], im)

# save figure
plt.savefig(os.path.join(result_folder,
                         f'{sub_id}_{task_name}_Fig03_Topomap_cnds.pdf'))
# ----------------------------------------------------------------------------

# # @@@ PLOT AVERAGE SNR CHANNELS @@@

fig, axs = plt.subplots(1, 3, figsize=(5, 4), sharey=True)

# all channels
fig.suptitle(f'Avg. SNR boost (Subject:{sub_id})')
axs[0].bar([1, 2], [boost_1f1_allCh, boost_1f2_allCh], color='grey')
axs[0].set(title='All channels', xticks=[1, 2],
           xticklabels=f_label, ylabel='SNR improvement')
cp.trim_axes(axs[0])
cp.prep4ai()

# occipital channels
axs[1].bar([1, 2], [boost_1f1_occCh, boost_1f2_occCh], color='grey')
axs[1].set(title='Occipital channels', xticks=[1, 2],
           xticklabels=f_label)
cp.trim_axes(axs[1])
cp.prep4ai()

# best channel
axs[2].bar([1, 2], [boost_1f1_bestCh, boost_1f2_bestCh], color='grey')
axs[2].set(title='Best channel', xticks=[1, 2],
           xticklabels=f_label)
cp.trim_axes(axs[2])
cp.prep4ai()

plt.tight_layout()

# save figure
plt.savefig(os.path.join(result_folder,
                         f'{sub_id}_{task_name}_Fig04_SNR_boost.pdf'))
