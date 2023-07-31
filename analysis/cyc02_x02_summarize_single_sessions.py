"""
Mohammad Shams <MShamsCBR@gmail.com>
June 22, 2023

In each of the four tasks, go through all the subjects, extract their
behavioral and eeg data, and save a summary for later population analysis.

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

task_names = ['oba_per', 'oba_cnt', 'fba_per', 'fba_cnt']
subject_ids = [5007, 5008, 5009, 5010, 5011, 5012, 5013, 5014, 5015]
task_number = 3

output_file_names = ['data_sum_oba_p.json', 'data_sum_oba_c.json',
                     'data_sum_fba_p.json', 'data_sum_fba_c.json']
task_name = task_names[task_number]
output_file_name = output_file_names[task_number]

df_pool = []

for isub, subject_id in enumerate(subject_ids):

    print(f'\n>>> Analysing Subject {subject_id} - Task {task_name} ...\n')

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
    # ---------------------------------------------------------------------

    # /// READ AND CONFIGURE BEHAVIORAL DATA ///

    # extract number of trials
    n_trials = beh_data.shape[0]

    # extract number of blocks
    n_blocks = beh_data['block_num'].max()
    n_trials_per_block = int(n_trials / n_blocks)

    # convert tilt magnitudes to degrees
    tilt_angle = (beh_data['tilt_magnitude'].values + 1) / 10

    # extract image-freq links
    # The first frequency in the 1 x 2 array in the 'Frequency column'
    # is for Blue
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

    # ---------------------------------------------------------------------

    # /// READ AND CONFIGURE EEG DATA ///

    eeg.info['line_freq'] = 60.
    # set montage
    montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
    eeg.set_montage(montage, match_alias=True)
    easycap_montage = mne.channels.make_standard_montage(
        'GSN-HydroCel-129')
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

    # /// calculate signal-to-noise ratio (SNR)
    tmin = tmin
    tmax = tmax
    fmin = 1.
    fmax = 50.
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
    # snrs: [64 trials] x [129 channels] x [115 freq. bins]
    snrs = snr_spectrum(psds, noise_n_neighbor_freqs=3,
                        noise_skip_neighbor_freqs=1)

    # define a freq range
    # freq_range: [103 freq. bins]
    freq_range = range(np.where(np.floor(freqs) == 1.)[0][0],
                       np.where(np.ceil(freqs) == fmax - 1)[0][0])

    # average PSD across all trials and all channels within the desired
    # freq range
    # psds_mean/std: [103 freq. bins]
    psds_plot = 10 * np.log10(psds)
    psds_mean = psds_plot.mean(axis=(0, 1))[freq_range]
    psds_std = psds_plot.std(axis=(0, 1))[freq_range]

    # /// prepare data for topography maps
    # find the closest frequency bin to stimulation frequency
    stim_freq1 = 7.5
    i_bin_1f1 = np.argmin(abs(freqs - stim_freq1 * 1))
    stim_freq2 = 12
    i_bin_1f2 = np.argmin(abs(freqs - stim_freq2 * 1))

    # index trials/events in a certain condition
    i_cnd1 = beh_data.index[beh_data['condition_num'] == 1]
    i_cnd2 = beh_data.index[beh_data['condition_num'] == 2]

    # indicate occipital channels
    occCh = ['E66', 'E69', 'E70', 'E71', 'E73', 'E74', 'E75',
             'E76', 'E81', 'E82', 'E83', 'E84', 'E88', 'E89']
    ind_occCh = np.nonzero(np.isin(epochs.info.ch_names, occCh))[0]

    # ---------------------------------------------------------------------

    # +++ TEST +++

    # make sure that number of events (in eeg file) matches the number
    # of trials
    # (in beh file)
    assert events.shape[0] == beh_data.shape[0]
    # ---------------------------------------------------------------------

    # @@@ SAVE BEHAVIORAL ANALYSES @@@
    cum_perf = beh_data['cummulative_performance'].iloc[-1]
    avg_rt = beh_data['avg_rt'].mean()
    tilt_angle = (beh_data['tilt_magnitude'].values + 1) / 10
    avg_tilt = tilt_angle.mean()

    # ---------------------------------------------------------------------

    # SNR spectrum
    freqs = freqs[freq_range]

    spec_cnd1_allCh = snrs[
        np.ix_(i_cnd1, range(snrs.shape[1]), freq_range)].mean(axis=(0, 1))
    spec_cnd2_allCh = snrs[
        np.ix_(i_cnd2, range(snrs.shape[1]), freq_range)].mean(axis=(0, 1))

    spec_cnd1_occCh = snrs[
        np.ix_(i_cnd1, ind_occCh, freq_range)].mean(axis=(0, 1))
    spec_cnd2_occCh = snrs[
        np.ix_(i_cnd2, ind_occCh, freq_range)].mean(axis=(0, 1))

    # -------------------------
    # topography maps

    topo_1f1 = snrs[:, :, i_bin_1f1].mean(axis=0)
    topo_1f2 = snrs[:, :, i_bin_1f2].mean(axis=0)

    topo_cnd1_1f1 = snrs[i_cnd1, :, i_bin_1f1].mean(axis=0)
    topo_cnd2_1f1 = snrs[i_cnd2, :, i_bin_1f1].mean(axis=0)

    topo_cnd1_1f2 = snrs[i_cnd1, :, i_bin_1f2].mean(axis=0)
    topo_cnd2_1f2 = snrs[i_cnd2, :, i_bin_1f2].mean(axis=0)

    topo_boost_1f1 = (topo_cnd1_1f1 - topo_cnd2_1f1) / topo_cnd2_1f1 * 100
    topo_boost_1f2 = (topo_cnd2_1f2 - topo_cnd1_1f2) / topo_cnd1_1f2 * 100

    # -------------------------
    # average boosts

    avg_boost_1f1_allCh = topo_boost_1f1.mean()
    avg_boost_1f2_allCh = topo_boost_1f2.mean()

    avg_boost_1f1_occCh = topo_boost_1f1[ind_occCh].mean()
    avg_boost_1f2_occCh = topo_boost_1f2[ind_occCh].mean()

    ind_bestCh_1f1 = np.argmax(topo_1f1[ind_occCh])
    ind_bestCh_1f2 = np.argmax(topo_1f2[ind_occCh])
    avg_boost_1f1_bestCh = topo_boost_1f1[ind_bestCh_1f1]
    avg_boost_1f2_bestCh = topo_boost_1f2[ind_bestCh_1f2]

    # -------------------------------------------------------------------------
    # /// create a dictionary of variables to be saved
    sub_dict = {
        'freq_label': [f_label],
        'cum_perf': [cum_perf],
        'avg_rt': [avg_rt],
        'avg_tilt': [avg_tilt],
        'frequencies': [freqs],
        'spec_cnd1_allCh': [spec_cnd1_allCh],
        'spec_cnd2_allCh': [spec_cnd2_allCh],
        'spec_cnd1_occCh': [spec_cnd1_occCh],
        'spec_cnd2_occCh': [spec_cnd2_occCh],
        'topo_1f1': [topo_1f1],
        'topo_1f2': [topo_1f2],
        'topo_cnd1_1f1': [topo_cnd1_1f1],
        'topo_cnd2_1f1': [topo_cnd2_1f1],
        'topo_cnd1_1f2': [topo_cnd1_1f2],
        'topo_cnd2_1f2': [topo_cnd2_1f2],
        'topo_boost_1f1': [topo_boost_1f1],
        'topo_boost_1f2': [topo_boost_1f2],
        'avg_boost_1f1_allCh': [avg_boost_1f1_allCh],
        'avg_boost_1f2_allCh': [avg_boost_1f2_allCh],
        'avg_boost_1f1_occCh': [avg_boost_1f1_occCh],
        'avg_boost_1f2_occCh': [avg_boost_1f2_occCh],
        'avg_boost_1f1_bestCh': [avg_boost_1f1_bestCh],
        'avg_boost_1f2_bestCh': [avg_boost_1f2_bestCh]
    }
    # convert to dict to data frame
    dfnew = pd.DataFrame(sub_dict)
    if isub == 0:
        df_pool = dfnew
        # if task_number == 0:
        #     epochs.save(os.path.join(data_folder, 'epochs-epo.fif'))
    else:
        df_pool = pd.concat([df_pool, dfnew], ignore_index=True)
    df_pool.to_json(os.path.join(data_folder, output_file_name))
