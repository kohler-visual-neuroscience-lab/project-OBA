import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cleanplot as cp
from scipy.stats import ttest_rel

"""
Mohammad Shams <mShamsResearch@gmail.com> Dec 28, 2022

Question: does each frequency tagged image appear contralaterally in the brain?
The data being tested here is from Subj 0002 while running exp01, recorded on 
Dec 12, 2022
"""


# /// functions
def snr_spectrum(psd, noise_n_neighbor_freqs=1, noise_skip_neighbor_freqs=1):
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


# /// load data
eeg_file = '0002_20221222_025047.mff'
beh_file = '0002_20221222_145047.json'
# set the full path to the raw data
eeg_path = os.path.join('..', 'data', 'rawData', 'test_exp01', eeg_file)
beh_path = os.path.join('..', 'data', 'rawData', 'test_exp01', beh_file)
raw = mne.io.read_raw_egi(eeg_path, preload=True)

# /// setup
raw.info['line_freq'] = 60.
# set montage
montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
raw.set_montage(montage, match_alias=True)
# set common average reference
raw.set_eeg_reference('average', projection=False, verbose=False)
# apply bandpass filter
raw.filter(l_freq=0.1, h_freq=None, fir_design='firwin', verbose=False)

# /// events
events = mne.find_events(raw, stim_channel=['CND1', 'CND2', 'CND3', 'CND4'])
event_id = raw.event_id
# visualize events
# fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'],
#                           first_samp=raw.first_samp, event_id=event_id)
# fig.subplots_adjust(right=0.7)  # make room for legend


# /// behavior
# read behavioral data
beh_data = pd.read_json(beh_path)
# +++ test +++
# make sure that number of events (in eeg file) matches the number of trials
# (in beh file)
assert events.shape[0] == beh_data.shape[0]
# ++++++++++++
n_trials = beh_data.shape[0]
# convert tilt magnitudes to degrees
tilt_angle = (beh_data['tilt_magnitude'].values + 1) / 10

# /// plot behavioral results
fig, axs = plt.subplots(3, figsize=(6, 6))

# @@@ plot performances over session
axs[0].plot(beh_data['cummulative_performance'], color='blue')
axs[0].plot(beh_data['running_performance'], color='tomato')
cp.trim_axes(axs[0], [0 - 2, n_trials], [0 - 5, 100 + 1])
axs[0].set_ylabel('Performance (%)')

# @@@ plot RT over session
axs[1].plot(beh_data['avg_rt'], 'o', markerfacecolor='k',
            markeredgecolor='none')
cp.trim_axes(axs[1], [0 - 2, n_trials], [0 - 50, 1000 + 1])
axs[1].set_ylabel('RT (ms)')

# @@@ plot tilt angles over session
axs[2].plot(tilt_angle, color='k')
cp.trim_axes(axs[2], [0 - 2, n_trials], [0 - .3, 5 + .1])
axs[2].set_ylabel('Tilt angle (deg)')
axs[2].set_xlabel('Trials')
plt.show()

# # times are wrt event times
# tmin = 0  # in sec
# tmax = 8  # in sec
# epochs = mne.Epochs(raw,
#                     events=events,
#                     event_id=[event_id['cnd1'], event_id['cnd2'],
#                               event_id['cnd3'], event_id['cnd4']],
#                     tmin=tmin, tmax=tmax, baseline=None, verbose=False)
# tmin = tmin
# tmax = tmax
# fmin = 1.
# fmax = 50.
# sfreq = epochs.info['sfreq']
# spectrum = epochs.compute_psd('welch', n_fft=int(sfreq * (tmax - tmin)),
#                               n_overlap=0, n_per_seg=None, tmin=tmin,
#                               tmax=tmax, fmin=fmin, fmax=fmax, window='boxcar',
#                               verbose=False)
# psds, freqs = spectrum.get_data(return_freqs=True)
# snrs = snr_spectrum(psds, noise_n_neighbor_freqs=3,
#                     noise_skip_neighbor_freqs=1)
#
# # find index of frequency bin closest to stimulation frequency
# stim_freq1 = 7.5
# stim_freq2 = 12
# i_bin_1f1 = np.argmin(abs(freqs - stim_freq1))
# i_bin_2f1 = np.argmin(abs(freqs - stim_freq1 * 2))
# i_bin_3f1 = np.argmin(abs(freqs - stim_freq1 * 3))
# i_bin_1f2 = np.argmin(abs(freqs - stim_freq2))
# i_bin_2f2 = np.argmin(abs(freqs - stim_freq2 * 2))
# i_bin_3f2 = np.argmin(abs(freqs - stim_freq2 * 3))
#
# i_trial_c1 = np.where(epochs.events[:, 2] == event_id['cnd1'])[0]
# i_trial_c2 = np.where(epochs.events[:, 2] == event_id['cnd2'])[0]
# i_trial_c3 = np.where(epochs.events[:, 2] == event_id['cnd3'])[0]
# i_trial_c4 = np.where(epochs.events[:, 2] == event_id['cnd4'])[0]
#
# # get average SNR at xx Hz for ALL channels
# freq = '1'
#
# cnd1_1 = snrs[i_trial_c1, :, i_bin_1f2].mean(axis=0)
# cnd2_1 = snrs[i_trial_c2, :, i_bin_1f2].mean(axis=0)
# cnd3_1 = snrs[i_trial_c3, :, i_bin_1f2].mean(axis=0)
# cnd4_1 = snrs[i_trial_c4, :, i_bin_1f2].mean(axis=0)
#
# cnd1_2 = snrs[i_trial_c1, :, i_bin_2f2].mean(axis=0)
# cnd2_2 = snrs[i_trial_c2, :, i_bin_2f2].mean(axis=0)
# cnd3_2 = snrs[i_trial_c3, :, i_bin_2f2].mean(axis=0)
# cnd4_2 = snrs[i_trial_c4, :, i_bin_2f2].mean(axis=0)
#
# cnd1_3 = snrs[i_trial_c1, :, i_bin_3f2].mean(axis=0)
# cnd2_3 = snrs[i_trial_c2, :, i_bin_3f2].mean(axis=0)
# cnd3_3 = snrs[i_trial_c3, :, i_bin_3f2].mean(axis=0)
# cnd4_3 = snrs[i_trial_c4, :, i_bin_3f2].mean(axis=0)
# # ==================================
# # plot SNR topography maps for the 1st harmonic
# # ==================================
# vlim1_13 = np.min([cnd1_1, cnd3_1])
# vlim2_13 = np.max([cnd1_1, cnd3_1])
# vlim1_24 = np.min([cnd2_1, cnd4_1])
# vlim2_24 = np.max([cnd2_1, cnd4_1])
# fig, ax = plt.subplots(2, 2)
# fig.suptitle(f"Subject: {data_path[0:4]}")
# ax[0, 0].set_title(f'FH_F_1f{freq}')
# ax[0, 1].set_title(f'HF_F_1f{freq}')
# ax[1, 0].set_title(f'FH_H_1f{freq}')
# ax[1, 1].set_title(f'HF_H_1f{freq}')
# mne.viz.plot_topomap(cnd1_1, epochs.info, axes=ax[0, 0],
#                      vlim=(vlim1_13, vlim2_13), show=False)
# mne.viz.plot_topomap(cnd2_1, epochs.info, axes=ax[0, 1],
#                      vlim=(vlim1_24, vlim2_24), show=False)
# mne.viz.plot_topomap(cnd3_1, epochs.info, axes=ax[1, 0],
#                      vlim=(vlim1_13, vlim2_13), show=False)
# mne.viz.plot_topomap(cnd4_1, epochs.info, axes=ax[1, 1],
#                      vlim=(vlim1_24, vlim2_24), show=False)
# plt.savefig(f'topomap_1f{freq}_{data_path[0:4]}.png')
# # ==================================
# # plot SNR topography maps for the 2nd harmonic
# # ==================================
# vlim1_13 = np.min([cnd1_2, cnd3_2])
# vlim2_13 = np.max([cnd1_2, cnd3_2])
# vlim1_24 = np.min([cnd2_2, cnd4_2])
# vlim2_24 = np.max([cnd2_2, cnd4_2])
# fig, ax = plt.subplots(2, 2)
# fig.suptitle(f"Subject: {data_path[0:4]}")
# ax[0, 0].set_title(f'FH_F_2f{freq}')
# ax[0, 1].set_title(f'HF_F_2f{freq}')
# ax[1, 0].set_title(f'FH_H_2f{freq}')
# ax[1, 1].set_title(f'HF_H_2f{freq}')
# mne.viz.plot_topomap(cnd1_2, epochs.info, axes=ax[0, 0],
#                      vlim=(vlim1_13, vlim2_13), show=False)
# mne.viz.plot_topomap(cnd2_2, epochs.info, axes=ax[0, 1],
#                      vlim=(vlim1_24, vlim2_24), show=False)
# mne.viz.plot_topomap(cnd3_2, epochs.info, axes=ax[1, 0],
#                      vlim=(vlim1_13, vlim2_13), show=False)
# mne.viz.plot_topomap(cnd4_2, epochs.info, axes=ax[1, 1],
#                      vlim=(vlim1_24, vlim2_24), show=False)
# plt.savefig(f'topomap_2f{freq}_{data_path[0:4]}.png')
# # ==================================
# # plot SNR topography maps for the 3rd harmonic
# # ==================================
# vlim1_13 = np.min([cnd1_3, cnd3_3])
# vlim2_13 = np.max([cnd1_3, cnd3_3])
# vlim1_24 = np.min([cnd2_3, cnd4_3])
# vlim2_24 = np.max([cnd2_3, cnd4_3])
# fig, ax = plt.subplots(2, 2)
# fig.suptitle(f"Subject: {data_path[0:4]}")
# ax[0, 0].set_title(f'FH_F_3f{freq}')
# ax[0, 1].set_title(f'HF_F_3f{freq}')
# ax[1, 0].set_title(f'FH_H_3f{freq}')
# ax[1, 1].set_title(f'HF_H_3f{freq}')
# mne.viz.plot_topomap(cnd1_3, epochs.info, axes=ax[0, 0],
#                      vlim=(vlim1_13, vlim2_13), show=False)
# mne.viz.plot_topomap(cnd2_3, epochs.info, axes=ax[0, 1],
#                      vlim=(vlim1_24, vlim2_24), show=False)
# mne.viz.plot_topomap(cnd3_3, epochs.info, axes=ax[1, 0],
#                      vlim=(vlim1_13, vlim2_13), show=False)
# mne.viz.plot_topomap(cnd4_3, epochs.info, axes=ax[1, 1],
#                      vlim=(vlim1_24, vlim2_24), show=False)
# plt.savefig(f'topomap_3f{freq}_{data_path[0:4]}.png')
# # ==================================
# # plot SNR difference maps for the first harmonic
# # ==================================
# if freq == '1':
#     face_boost = cnd1_1 - cnd3_1
#     house_boost = cnd4_1 - cnd2_1
# elif freq == '2':
#     face_boost = cnd2_1 - cnd4_1
#     house_boost = cnd3_1 - cnd1_1
# else:
#     print("### Requested frequency is out of range. ###")
#     face_boost = None
#     house_boost = None
#
# vlim1_F = np.min(face_boost)
# vlim2_F = np.max(face_boost)
# vlim1_H = np.min(house_boost)
# vlim2_H = np.max(house_boost)
# fig, ax = plt.subplots(1, 2)
# fig.suptitle(f"Subject: {data_path[0:4]}")
# ax[0].set_title(f'Face_boost_1f{freq}')
# ax[1].set_title(f'House_boost_1f{freq}')
# mne.viz.plot_topomap(face_boost, epochs.info, axes=ax[0], show=False,
#                      vlim=(vlim1_F, vlim2_F))
# mne.viz.plot_topomap(house_boost, epochs.info, axes=ax[1], show=False,
#                      vlim=(vlim1_H, vlim2_H))
# plt.savefig(f'topomap_diff_1f{freq}_{data_path[0:4]}.png')
# # ==================================
# # plot SNR difference maps (rms average across harmonics)
# # ==================================
# cnd1_rms = np.empty(len(cnd1_1))
# cnd2_rms = np.empty(len(cnd1_1))
# cnd3_rms = np.empty(len(cnd1_1))
# cnd4_rms = np.empty(len(cnd1_1))
# cnd1_rms[:] = np.nan
# cnd2_rms[:] = np.nan
# cnd3_rms[:] = np.nan
# cnd4_rms[:] = np.nan
# for i in range(len(cnd1_1)):
#     cnd1_rms[i] = np.sqrt(np.mean(np.power([cnd1_1[i], cnd1_2[i], cnd1_3[i]],
#                                            2)))
#     cnd2_rms[i] = np.sqrt(np.mean(np.power([cnd2_1[i], cnd2_2[i], cnd2_3[i]],
#                                            2)))
#     cnd3_rms[i] = np.sqrt(np.mean(np.power([cnd3_1[i], cnd3_2[i], cnd3_3[i]],
#                                            2)))
#     cnd4_rms[i] = np.sqrt(np.mean(np.power([cnd4_1[i], cnd4_2[i], cnd4_3[i]],
#                                            2)))
# if freq == '1':
#     face_boost = cnd1_rms - cnd3_rms
#     house_boost = cnd4_rms - cnd2_rms
# elif freq == '2':
#     face_boost = cnd2_rms - cnd4_rms
#     house_boost = cnd3_rms - cnd1_rms
# else:
#     print("### Requested frequency is out of range. ###")
#     face_boost = None
#     house_boost = None
#
# vlim1_F = np.min(face_boost)
# vlim2_F = np.max(face_boost)
# vlim1_H = np.min(house_boost)
# vlim2_H = np.max(house_boost)
#
# fig, ax = plt.subplots(1, 2)
# fig.suptitle(f"Subject: {data_path[0:4]}")
# ax[0].set_title(f'Face_boost_f{freq}')
# ax[1].set_title(f'House_boost_f{freq}')
# mne.viz.plot_topomap(face_boost, epochs.info, axes=ax[0], show=False,
#                      vlim=(vlim1_F, vlim2_F))
# mne.viz.plot_topomap(house_boost, epochs.info, axes=ax[1], show=False,
#                      vlim=(vlim1_H, vlim2_H))
# plt.savefig(f'topomap_diff_123f{freq}_{data_path[0:4]}.png')
