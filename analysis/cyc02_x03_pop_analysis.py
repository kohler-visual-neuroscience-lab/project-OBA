"""
Mohammad Shams <MoShamsCBR@gmail.com>
June 22, 2023

Population analysis based on summarized data.

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib import cleanplot as cp


def clean_bar(ax):
    cp.trim_axes(ax)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', length=0)


task_number = 0
output_file_names = ['data_sum_oba_p.json', 'data_sum_oba_c.json',
                     'data_sum_fba_p.json', 'data_sum_fba_c.json']
task_show_names = ['OBA-periphery', 'OBA-center',
                   'FBA-periphery', 'FBA-center']
output_file_name = output_file_names[task_number]

data_folder = os.path.join('..', 'data', 'cycle02')
result_folder = os.path.join('..', 'result', 'cycle02')
df_pool = pd.read_json(os.path.join(data_folder, output_file_name))

nsub = df_pool.shape[0]

# -----------------------------------------------------------------------------
# @@@ PLOT BEHAVIORAL ANALYSES @@@

fig, axs = plt.subplots(1, 3, figsize=(5, 4))
fig.suptitle(f'Pop. Behavior - {task_show_names[task_number]} (N={nsub})')
cp.prep4ai()

color = 'silver'

axs[0].bar(1, df_pool.cum_perf.mean(), color=color)
axs[0].plot(np.ones(nsub), df_pool.cum_perf, 'o', mec='black', mfc='none')
axs[0].set(ylabel='Average Performance [%]', xticks=[])
clean_bar(axs[0])

axs[1].bar(1, df_pool.avg_tilt.mean(), color=color)
axs[1].plot(np.ones(nsub), df_pool.avg_tilt, 'o', mec='black', mfc='none')
axs[1].set(ylabel='Average Tilt Angle [deg]', xticks=[])
clean_bar(axs[1])

axs[2].bar(1, df_pool.avg_rt.mean(), color=color)
axs[2].plot(np.ones(nsub), df_pool.avg_rt, 'o', mec='black', mfc='none')
axs[2].set(ylabel='Average RT [ms]', xticks=[])
clean_bar(axs[2])

plt.tight_layout()
plt.savefig(os.path.join(result_folder,
                         f'Pop_{task_show_names[task_number]}_Fig01_Behavior.pdf'))
# ---------------------------------------------------------------------
#
# # @@@ PLOT PSD, SNR, 1F1, 1F2 @@@
#
# fig, axes = plt.subplots(2, 2, figsize=(10, 6), width_ratios=[3, 1])
# fig.suptitle(f'SNR and Topo-map (Subject:{sub_id}-Task:{task_name})')
# cp.prep4ai()
#
# # -------------------------
# # SNR spectrum (avg. across all channels)
#
# fmin_show = 5
# fmax_show = 15
# fstep = 1
#
# spec_cnd1_allCh = snrs[
#     np.ix_(i_cnd1, range(snrs.shape[1]), freq_range)].mean(axis=(0, 1))
# spec_cnd2_allCh = snrs[
#     np.ix_(i_cnd2, range(snrs.shape[1]), freq_range)].mean(axis=(0, 1))
#
# axes[0, 0].plot(freqs[freq_range], spec_cnd1_allCh, color='darkcyan',
#                 label='cnd1-allCh', zorder=3)
# axes[0, 0].plot(freqs[freq_range], spec_cnd2_allCh, color='darkorchid',
#                 label='cnd2-allCh', zorder=4)
# axes[0, 0].set(ylabel='SNR',
#                xticks=np.arange(fmin_show, fmax_show + fstep, fstep),
#                xlim=[fmin_show, fmax_show])
# axes[0, 0].axvline(x=7.5, color='silver', zorder=1)
# axes[0, 0].axvline(x=12, color='silver', zorder=2)
# leg = axes[0, 0].legend()
# leg.get_frame().set_linewidth(0)
# cp.trim_axes(axes[0, 0])
#
# # -------------------------
# # SNR spectrum (avg. across occipital channels)
#
# spec_cnd1_occCh = snrs[
#     np.ix_(i_cnd1, ind_occCh, freq_range)].mean(axis=(0, 1))
# spec_cnd2_occCh = snrs[
#     np.ix_(i_cnd2, ind_occCh, freq_range)].mean(axis=(0, 1))
#
# axes[1, 0].plot(freqs[freq_range], spec_cnd1_occCh, color='darkcyan',
#                 label='cnd1-occCh', zorder=3)
# axes[1, 0].plot(freqs[freq_range], spec_cnd2_occCh, color='darkorchid',
#                 label='cnd2-occCh', zorder=4)
# axes[1, 0].set(xlabel='Frequency [Hz]', ylabel='SNR',
#                xticks=np.arange(fmin_show, fmax_show + fstep, fstep),
#                xlim=[fmin_show, fmax_show])
# axes[1, 0].axvline(x=7.5, color='silver', zorder=1)
# axes[1, 0].axvline(x=12, color='silver', zorder=2)
# leg = axes[1, 0].legend()
# leg.get_frame().set_linewidth(0)
# cp.trim_axes(axes[1, 0])
#
# # -------------------------
# # topography maps
#
# topo_1f1 = snrs[:, :, i_bin_1f1].mean(axis=0)
# topo_1f2 = snrs[:, :, i_bin_1f2].mean(axis=0)
#
# vmin = 1
# vmax = np.max([topo_1f1, topo_1f2])
#
# im, _ = mne.viz.plot_topomap(topo_1f1, epochs.info, vlim=(vmin, vmax),
#                              axes=axes[0, 1], show=False)
# axes[0, 1].set(title='f1')
# cp.add_snr_colorbar(fig, axes[0, 1], im)
#
# im, _ = mne.viz.plot_topomap(topo_1f2, epochs.info, vlim=(vmin, vmax),
#                              axes=axes[1, 1], show=False)
# axes[1, 1].set(title='f2')
# cp.add_snr_colorbar(fig, axes[1, 1], im)
#
# # save figure
# plt.savefig(os.path.join(result_folder,
#                          f'{sub_id}_{task_name}_Fig02_SNR_Topomap.pdf'))
# # ---------------------------------------------------------------------
#
# # @@@ PLOT TOPOGRAPHY MAPS FOR EACH CONDITION AT EACH FREQUENCY @@@
#
# # set up the min and max of the color map (insert 'None' to pass)
# fig, axes = plt.subplots(2, 3, figsize=(7.5, 6))
# fig.suptitle(f'Topography map at each condition and '
#              f'each frequency (Subject:{sub_id}-Task:{task_name})')
# cp.prep4ai()
#
# # -------------------------
# # @@@ f1 boost pairs
#
# topo_cnd1_1f1 = snrs[i_cnd1, :, i_bin_1f1].mean(axis=0)
# topo_cnd2_1f1 = snrs[i_cnd2, :, i_bin_1f1].mean(axis=0)
#
# # Topography map of CND1 1f1:
# im, _ = mne.viz.plot_topomap(topo_cnd1_1f1, epochs.info,
#                              axes=axes[0, 0],
#                              vlim=(vmin, vmax), show=False)
# axes[0, 0].set(title='CND1 f1')
# cp.add_snr_colorbar(fig, axes[0, 0], im)
# # Topography map of CND1 1f2:
# im, _ = mne.viz.plot_topomap(topo_cnd2_1f1, epochs.info,
#                              axes=axes[0, 1],
#                              vlim=(vmin, vmax), show=False)
# axes[0, 1].set(title='CND2 f1')
# cp.add_snr_colorbar(fig, axes[0, 1], im)
#
# # -------------------------
# # @@@ f2 boost pairs
#
# topo_cnd1_1f2 = snrs[i_cnd1, :, i_bin_1f2].mean(axis=0)
# topo_cnd2_1f2 = snrs[i_cnd2, :, i_bin_1f2].mean(axis=0)
#
# # Topography map of CND2 1f2:
# im, _ = mne.viz.plot_topomap(topo_cnd2_1f2, epochs.info,
#                              axes=axes[1, 0],
#                              vlim=(vmin, vmax), show=False)
# axes[1, 0].set(title='CND2 f2')
# cp.add_snr_colorbar(fig, axes[1, 0], im)
# # Topography map of CND2, 1f1:
# im, _ = mne.viz.plot_topomap(topo_cnd1_1f2, epochs.info,
#                              axes=axes[1, 1],
#                              vlim=(vmin, vmax), show=False)
# axes[1, 1].set(title='CND1 f2')
# cp.add_snr_colorbar(fig, axes[1, 1], im)
#
# # -------------------------
# # @@@ ADD OBJECT BOOST MAPS
#
# topo_boost_1f1 = topo_cnd1_1f1 - topo_cnd2_1f1
# topo_boost_1f2 = topo_cnd2_1f2 - topo_cnd1_1f2
#
# # set up the min and max of the color map (insert 'None' to pass)
# vmin_diff = -vmax
# vmax_diff = vmax
# # Topography map of 1f1 boost:
# im, _ = mne.viz.plot_topomap(topo_boost_1f1, epochs.info,
#                              axes=axes[0, 2],
#                              vlim=(vmin_diff, vmax_diff), show=False)
# axes[0, 2].set(title=f'f1 boost ({f_label[0]})')
# cp.add_snr_colorbar(fig, axes[0, 2], im)
# # Topography map of 1f2 boost:
# im, _ = mne.viz.plot_topomap(topo_boost_1f2, epochs.info,
#                              axes=axes[1, 2],
#                              vlim=(vmin_diff, vmax_diff), show=False)
# axes[1, 2].set(title=f'f2 boost ({f_label[1]})')
# cp.add_snr_colorbar(fig, axes[1, 2], im)
#
# # save figure
# plt.savefig(os.path.join(result_folder,
#                          f'{sub_id}_{task_name}_Fig03_Topomap_cnds.pdf'))
# # ---------------------------------------------------------------------
#
# # # @@@ PLOT AVERAGE SNR CHANNELS @@@
#
# fig, axs = plt.subplots(1, 3, figsize=(5, 4), sharey=True)
#
# # ---------------------------------
# # all channels
# trials_cnd1_1f1 = snrs[i_cnd1, :, i_bin_1f1].mean(axis=1)
# trials_cnd2_1f1 = snrs[i_cnd2, :, i_bin_1f1].mean(axis=1)
# trials_cnd1_1f2 = snrs[i_cnd1, :, i_bin_1f2].mean(axis=1)
# trials_cnd2_1f2 = snrs[i_cnd2, :, i_bin_1f2].mean(axis=1)
#
# trials_boost_1f1 = trials_cnd1_1f1 - trials_cnd2_1f1
# trials_boost_1f2 = trials_cnd2_1f2 - trials_cnd1_1f2
#
# fig.suptitle(f'Avg. SNR boost (Subject:{sub_id}-Task:{task_name})')
# axs[0].bar([1, 2], [trials_boost_1f1.mean(), trials_boost_1f2.mean()],
#            color='silver')
# axs[0].set(title='All channels', xticks=[1, 2],
#            xticklabels=f_label, ylabel='SNR improvement')
#
# axs[0].errorbar([1, 2],
#                 [trials_boost_1f1.mean(), trials_boost_1f2.mean()],
#                 yerr=[trials_boost_1f1.std() / (n_trials ** .5),
#                       trials_boost_1f2.std() / (n_trials ** .5)],
#                 color='black', fmt='none')
#
# cp.trim_axes(axs[0])
# cp.prep4ai()
# # ---------------------------------
# # occipital channels
# trials_cnd1_1f1 = snrs[np.ix_(i_cnd1, ind_occCh, [i_bin_1f1])].mean(
#     axis=1)
# trials_cnd2_1f1 = snrs[np.ix_(i_cnd2, ind_occCh, [i_bin_1f1])].mean(
#     axis=1)
# trials_cnd1_1f2 = snrs[np.ix_(i_cnd1, ind_occCh, [i_bin_1f2])].mean(
#     axis=1)
# trials_cnd2_1f2 = snrs[np.ix_(i_cnd2, ind_occCh, [i_bin_1f2])].mean(
#     axis=1)
#
# trials_boost_1f1 = trials_cnd1_1f1 - trials_cnd2_1f1
# trials_boost_1f2 = trials_cnd2_1f2 - trials_cnd1_1f2
#
# axs[1].bar([1, 2], [trials_boost_1f1.mean(), trials_boost_1f2.mean()],
#            color='silver')
# axs[1].set(title='Occipital channels', xticks=[1, 2],
#            xticklabels=f_label)
#
# axs[1].errorbar([1, 2],
#                 [trials_boost_1f1.mean(), trials_boost_1f2.mean()],
#                 yerr=[trials_boost_1f1.std() / (n_trials ** .5),
#                       trials_boost_1f2.std() / (n_trials ** .5)],
#                 color='black', fmt='none')
#
# cp.trim_axes(axs[1])
# cp.prep4ai()
# # ---------------------------------
# # best channel
# ind_bestCh_1f1 = np.argmax(topo_1f1[ind_occCh])
# ind_bestCh_1f2 = np.argmax(topo_1f2[ind_occCh])
#
# axs[2].bar([1, 2], [trials_boost_1f1[ind_bestCh_1f1][0],
#                     trials_boost_1f2[ind_bestCh_1f2][0]],
#            color='silver')
# axs[2].set(title='Best channel', xticks=[1, 2],
#            xticklabels=f_label)
# cp.trim_axes(axs[2])
# cp.prep4ai()
#
# plt.tight_layout()
#
# # save figure
# plt.savefig(os.path.join(result_folder,
#                          f'{sub_id}_{task_name}_Fig04_SNR_boost.pdf'))
#
# print('Done.\n\n')
