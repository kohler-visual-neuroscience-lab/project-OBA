import os
import mne
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib import cleanplot as cp

"""
MoShams <MShamsCBR@gmail.com> Jan 23, 2023

Pools all the EEG analyses done for individuals and generates the population
result.

"""


# /// FUNCTIONS
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


def clean_bar(ax):
    cp.trim_axes(ax)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', length=0)


# ----------------------------------------------------------------------------

# /// SET UP SOURCE DATA PATH AND PARAMETERS ///

eeg_file = []
beh_file = []
df_pool = []

eeg_file.append('0005_20230113_105012.mff')
beh_file.append('0005_20230113_104958.json')
eeg_file.append('0010_20230113_010843.mff')
beh_file.append('0010_20230113_130828.json')
eeg_file.append('0001_20230120_120918.mff')
beh_file.append('0001_20230120_120903.json')
eeg_file.append('0008_20230120_023153.mff')
beh_file.append('0008_20230120_143138.json')
eeg_file.append('0009_20230120_105003.mff')
beh_file.append('0009_20230120_104948.json')
eeg_file.append('0011_20230120_012119.mff')
beh_file.append('0011_20230120_132104.json')
eeg_file.append('0004_20230329_123905.mff')
beh_file.append('0004_20230329_113904.json')

# ----------------------------------------------------------------------------
# +++ TEST +++
nsub1 = len(eeg_file)
nsub2 = len(beh_file)
assert nsub1 == nsub2
# ++++++++++++
nsub = nsub1

for isub in range(np.size(beh_file)):
    # set the full path to the raw data
    eeg_path = os.path.join('..', 'data', 'exp01_v01', 'raw', eeg_file[isub])
    beh_path = os.path.join('..', 'data', 'exp01_v01', 'raw', beh_file[isub])
    # +++ TEST +++
    assert eeg_file[isub][:13] == beh_file[isub][:13]
    # ++++++++++++
    print(f"\n\n*** Analyzing subject #{isub + 1}...")
    eeg = mne.io.read_raw_egi(eeg_path, preload=True)
    beh = pd.read_json(beh_path)

    # -------------------------------

    # /// SET UP SAVE PATH AND PARAMETERS ///

    save_path = os.path.join('..', 'result', 'exp01_v01')
    # extract subject's ID
    sub_id = beh_file[:4]
    # extract recoding date
    rec_date = beh_file[5:13]
    # -------------------------------

    # /// read and configure behavioral data
    # set number of blocks
    n_blocks = 4
    # extract number of trials
    n_trials = beh.shape[0]
    # convert tilt magnitudes to degrees
    tilt_angle = (beh['tilt_magnitude'].values + 1) / 10

    # /// average across trials
    # performance
    cum_perf = beh['cummulative_performance'].mean()
    run_perf = beh['running_performance'].mean()
    # RT
    rt = beh['avg_rt'].mean()
    # tilt angles
    tilt = tilt_angle.mean()

    # -------------------------------

    # /// read and configure eeg data
    eeg.info['line_freq'] = 60.
    # set montage
    montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
    eeg.set_montage(montage, match_alias=True)
    # set common average reference
    eeg.set_eeg_reference('average', projection=False, verbose=False)
    # apply bandpass filter
    eeg.filter(l_freq=0.1, h_freq=None, fir_design='firwin', verbose=False)

    # /// extract events
    events = mne.find_events(eeg,
                             stim_channel=['CND1', 'CND2', 'CND3', 'CND4'])
    event_id = eeg.event_id

    # /// reshape data into epochs
    tmin = 0  # in sec wrt event times
    tmax = 10  # in sec wrt event times
    epochs = mne.Epochs(eeg, events=events,
                        event_id=[event_id['CND1'], event_id['CND2'],
                                  event_id['CND3'], event_id['CND4']],
                        tmin=tmin, tmax=tmax, baseline=None, verbose=False)

    # /// calculate SNR and PSD
    tmin = tmin
    tmax = tmax
    fmin = 1.
    fmax = 30.
    sampling_freq = epochs.info['sfreq']
    # calculate PSD
    spectrum = epochs.compute_psd('welch',
                                  n_fft=int(sampling_freq * (tmax - tmin)),
                                  n_overlap=0, n_per_seg=None, tmin=tmin,
                                  tmax=tmax, fmin=fmin, fmax=fmax,
                                  window='boxcar',
                                  verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)
    # calculate SNR
    snrs = snr_spectrum(psds, noise_n_neighbor_freqs=3,
                        noise_skip_neighbor_freqs=1)
    # define a freq range
    freq_range = range(np.where(np.floor(freqs) == 1.)[0][0],
                       np.where(np.ceil(freqs) == fmax - 1)[0][0])
    # average PSD across all trials and channels within the desired freq range
    psds_plot = 10 * np.log10(psds)
    psds_mean = psds_plot.mean(axis=(0, 1))[freq_range]
    psds_std = psds_plot.std(axis=(0, 1))[freq_range]
    # average SNR across all trials and channels within the desired freq range
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
    # i_cnd1 = events[:, 2] == 1
    # i_cnd2 = events[:, 2] == 2
    # i_cnd3 = events[:, 2] == 3
    # i_cnd4 = events[:, 2] == 4
    i_cnd1 = beh.condition_num == 1
    i_cnd2 = beh.condition_num == 2
    i_cnd3 = beh.condition_num == 3
    i_cnd4 = beh.condition_num == 4

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

    # for each image at each frequency, subtract unattended map from
    # attended to obtain "image boost" topography map
    face_boost_1f1 = (snrs_cnd1_1f1_avg - snrs_cnd3_1f1_avg) / \
                     snrs_cnd3_1f1_avg * 100
    face_boost_1f2 = (snrs_cnd2_1f2_avg - snrs_cnd4_1f2_avg) / \
                     snrs_cnd4_1f2_avg * 100
    house_boost_1f1 = (snrs_cnd4_1f1_avg - snrs_cnd2_1f1_avg) / \
                      snrs_cnd2_1f1_avg * 100
    house_boost_1f2 = (snrs_cnd3_1f2_avg - snrs_cnd1_1f2_avg) / \
                      snrs_cnd1_1f2_avg * 100

    # average across all channels
    face_boost = np.mean((face_boost_1f1 + face_boost_1f2) / 2)
    house_boost = np.mean((house_boost_1f1 + house_boost_1f2) / 2)

    # average across occipital channels
    occ_ch = ['E66', 'E69', 'E70', 'E71', 'E73', 'E74', 'E75',
              'E76', 'E81', 'E82', 'E83', 'E84', 'E88', 'E89']
    ind_occ = np.nonzero(np.isin(epochs.info.ch_names, occ_ch))
    face_boost_occ = np.mean(((face_boost_1f1 + face_boost_1f2) / 2)[ind_occ])
    house_boost_occ = np.mean(
        ((house_boost_1f1 + house_boost_1f2) / 2)[ind_occ])

    # -------------------------------

    # /// create a dictionary of variables to be saved
    sub_dict = {'cum_perf': [cum_perf],
                'rt': [rt],
                'tilt': [tilt],
                'face_boost': [face_boost],
                'house_boost': [house_boost],
                'face_boost_occ': [face_boost_occ],
                'house_boost_occ': [house_boost_occ]}
    # convert to dict to data frame
    dfnew = pd.DataFrame(sub_dict)
    if isub == 0:
        df_pool = dfnew
    else:
        df_pool = pd.concat([df_pool, dfnew], ignore_index=True)
# ----------------------------------------------------------------------------

# @@@ PLOT BEHAVIOR

fig, axs = plt.subplots(1, 3, figsize=(5, 4))
fig.suptitle(f'Population Behavior (N={nsub})')
axs[0].bar(1, df_pool.cum_perf.mean(), color='grey')
axs[0].plot(np.ones(nsub), df_pool.cum_perf, 'o', mec='black', mfc='none')
axs[0].set(ylabel='Average Performance [%]', xticks=[])
clean_bar(axs[0])

axs[1].bar(1, df_pool.tilt.mean(), color='grey')
axs[1].plot(np.ones(nsub), df_pool.tilt, 'o', mec='black', mfc='none')
axs[1].set(ylabel='Average Tilt Angle [deg]', xticks=[])
clean_bar(axs[1])

axs[2].bar(1, df_pool.rt.mean(), color='grey')
axs[2].plot(np.ones(nsub), df_pool.rt, 'o', mec='black', mfc='none')
axs[2].set(ylabel='Average RT [ms]', xticks=[])
clean_bar(axs[2])

plt.tight_layout()
plt.savefig(os.path.join(save_path, f'Pop_N{nsub}_Behavior.pdf'))
# ----------------------------------------------------------------------------

# @@@ PLOT EEG

# @ all channels
fig, axs = plt.subplots(1, 2, figsize=(5, 4), sharey=True)
fig.suptitle(f'Population SNR Improvement (N={nsub})')
axs[0].bar([1, 2], [df_pool.face_boost.mean(),
                    df_pool.house_boost.mean()], color='grey')
# add individuals
axs[0].plot(np.ones(nsub), df_pool.face_boost,
            'o', mec='black', mfc='none')
axs[0].plot(2 * np.ones(nsub), df_pool.house_boost,
            'o', mec='black', mfc='none')
# connect face and house improvement of each individual
xxs = np.tile([1, 2], (nsub, 1)).transpose()
yys = np.vstack((df_pool.face_boost, df_pool.house_boost))
axs[0].plot(xxs, yys, color='black', lw=.5)
# add-ons
axs[0].set(xticks=[1, 2], xticklabels=['Face', 'House'],
           title='All Channels', ylabel='SNR Improvement [%]')
clean_bar(axs[0])

# @ occipital channels
axs[1].bar([1, 2], [df_pool.face_boost_occ.mean(),
                    df_pool.house_boost_occ.mean()], color='grey')
# add individuals
axs[1].plot(np.ones(nsub), df_pool.face_boost_occ,
            'o', mec='black', mfc='none')
axs[1].plot(2 * np.ones(nsub), df_pool.house_boost_occ,
            'o', mec='black', mfc='none')
# connect face and house improvement of each individual
xxs = np.tile([1, 2], (nsub, 1)).transpose()
yys = np.vstack((df_pool.face_boost_occ, df_pool.house_boost_occ))
axs[1].plot(xxs, yys, color='black', lw=.5)
axs[1].set(xticks=[1, 2], xticklabels=['Face', 'House'],
           title='Occipital Channels', ylabel='SNR Improvement [%]')
clean_bar(axs[1])

plt.tight_layout()
plt.savefig(os.path.join(save_path, f'Pop_N{nsub}_SNR.pdf'))

# ----------------------------------------------------------------------------
# stats
stat_face = scipy.stats.wilcoxon(df_pool.face_boost_occ)
stat_house = scipy.stats.wilcoxon(df_pool.house_boost_occ)
print(f'Face mean improv. = {df_pool.face_boost_occ.mean()}')
print(f'p-value = {stat_face.pvalue}')
print(f'House mean improv. = {df_pool.house_boost_occ.mean()}')
print(f'p-value = {stat_house.pvalue}')

fig, axs = plt.subplots(1, 2)
axs[0].scatter(df_pool.tilt, df_pool.face_boost_occ)
axs[1].scatter(df_pool.tilt, df_pool.house_boost_occ)
