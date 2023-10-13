"""
Mohammad Shams <MShamsCBR@gmail.com>
June 22, 2023

Population analysis based on summarized data.

"""

import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib import cleanplot as cp


def clean_bar(ax):
    cp.trim_axes(ax)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', length=0)


def add_snr_colorbar(fig1, ax, im1):
    cbar_ax = ax.inset_axes([1, .05, .04, .85])
    fig1.colorbar(im1, cax=cbar_ax)
    cbar_ax.set(title='SNR (norm)')


task_number = 3
output_file_names = ['data_sum_oba_p.json', 'data_sum_oba_c.json',
                     'data_sum_fba_p.json', 'data_sum_fba_c.json']
task_show_names = ['OBA-periphery', 'OBA-center',
                   'FBA-periphery', 'FBA-center']
output_file_name = output_file_names[task_number]

data_folder = os.path.join('..', 'data', 'cycle02')
result_folder = os.path.join('..', 'result', 'cycle02')
df_pool = pd.read_json(os.path.join(data_folder, output_file_name))

epochs = mne.read_epochs(os.path.join(data_folder, 'epochs-epo.fif'))

f_label = df_pool['freq_label'][0]

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
                         f'Pop_{task_show_names[task_number]}_'
                         f'Fig01_behavior.pdf'))

# ---------------------------------------------------------------------
# @@@ PLOT SNR, 1F1, 1F2 @@@

freqs = df_pool['frequencies'][0]

arr = np.array(df_pool['spec_cnd1_allCh'].tolist())
arr = np.array(
    [[np.nan if val is None else val for val in row] for row in arr])
spec_cnd1_allCh = arr.mean(axis=0)

arr = np.array(df_pool['spec_cnd2_allCh'].tolist())
arr = np.array(
    [[np.nan if val is None else val for val in row] for row in arr])
spec_cnd2_allCh = arr.mean(axis=0)

arr = np.array(df_pool['spec_cnd1_occCh'].tolist())
arr = np.array(
    [[np.nan if val is None else val for val in row] for row in arr])
spec_cnd1_occCh = arr.mean(axis=0)

arr = np.array(df_pool['spec_cnd2_occCh'].tolist())
arr = np.array(
    [[np.nan if val is None else val for val in row] for row in arr])
spec_cnd2_occCh = arr.mean(axis=0)

fig, axes = plt.subplots(2, 2, figsize=(10, 6), width_ratios=[3, 1])
fig.suptitle(f'Pop. SNR & Topomap - {task_show_names[task_number]} (N={nsub})')
cp.prep4ai()

# -------------------------
# SNR spectrum (avg. across all channels)

fmin_show = 5
fmax_show = 15
fstep = 1

axes[0, 0].plot(freqs, spec_cnd1_allCh, color='darkcyan',
                label='cnd1-allCh', zorder=3)
axes[0, 0].plot(freqs, spec_cnd2_allCh, color='darkorchid',
                label='cnd2-allCh', zorder=4)
axes[0, 0].set(ylabel='SNR',
               xticks=np.arange(fmin_show, fmax_show + fstep, fstep),
               xlim=[fmin_show, fmax_show])
axes[0, 0].axvline(x=7.5, color='silver', zorder=1)
axes[0, 0].axvline(x=12, color='silver', zorder=2)
leg = axes[0, 0].legend()
leg.get_frame().set_linewidth(0)
cp.trim_axes(axes[0, 0])

# -------------------------
# SNR spectrum (avg. across occipital channels)

axes[1, 0].plot(freqs, spec_cnd1_occCh, color='darkcyan',
                label='cnd1-occCh', zorder=3)
axes[1, 0].plot(freqs, spec_cnd2_occCh, color='darkorchid',
                label='cnd2-occCh', zorder=4)
axes[1, 0].set(xlabel='Frequency [Hz]', ylabel='SNR',
               xticks=np.arange(fmin_show, fmax_show + fstep, fstep),
               xlim=[fmin_show, fmax_show])
axes[1, 0].axvline(x=7.5, color='silver', zorder=1)
axes[1, 0].axvline(x=12, color='silver', zorder=2)
leg = axes[1, 0].legend()
leg.get_frame().set_linewidth(0)
cp.trim_axes(axes[1, 0])

# -------------------------
# topography maps

topo_1f1 = np.array(df_pool['topo_1f1'].tolist())
topo_1f2 = np.array(df_pool['topo_1f2'].tolist())
topo_1f1 = np.mean(topo_1f1 / np.max(topo_1f1, axis=1, keepdims=True),
                   axis=0)
topo_1f2 = np.mean(topo_1f2 / np.max(topo_1f2, axis=1, keepdims=True),
                   axis=0)

vmin = np.min([topo_1f1, topo_1f2])
vmax = np.max([topo_1f1, topo_1f2])

im, _ = mne.viz.plot_topomap(topo_1f1, epochs.info, vlim=(vmin, vmax),
                             axes=axes[0, 1], show=False)
axes[0, 1].set(title='f1')
add_snr_colorbar(fig, axes[0, 1], im)

im, _ = mne.viz.plot_topomap(topo_1f2, epochs.info, vlim=(vmin, vmax),
                             axes=axes[1, 1], show=False)
axes[1, 1].set(title='f2')
add_snr_colorbar(fig, axes[1, 1], im)

# save figure
plt.savefig(os.path.join(result_folder,
                         f'Pop_{task_show_names[task_number]}_'
                         f'Fig02_SNR_Topomap.pdf'))
# ---------------------------------------------------------------------

# @@@ PLOT TOPOGRAPHY MAPS FOR EACH CONDITION AT EACH FREQUENCY @@@

# set up the min and max of the color map (insert 'None' to pass)
fig, axes = plt.subplots(2, 3, figsize=(7.5, 6))
fig.suptitle(f'Pop. CND-based Topomaps - {task_show_names[task_number]} '
             f'(N={nsub})')
cp.prep4ai()

# -------------------------
# @@@ f1 boost pairs

topo_cnd1_1f1 = np.array(df_pool['topo_cnd1_1f1'].tolist())
topo_cnd2_1f1 = np.array(df_pool['topo_cnd2_1f1'].tolist())

topo_cnd1_1f1 = np.mean(topo_cnd1_1f1 / np.max(topo_cnd1_1f1,
                                               axis=1, keepdims=True), axis=0)
topo_cnd2_1f1 = np.mean(topo_cnd2_1f1 / np.max(topo_cnd2_1f1,
                                               axis=1, keepdims=True), axis=0)

vmin = np.min([topo_cnd1_1f1, topo_cnd2_1f1])
vmax = np.max([topo_cnd1_1f1, topo_cnd2_1f1])

# Topography map of CND1 1f1:
im, _ = mne.viz.plot_topomap(topo_cnd1_1f1, epochs.info,
                             axes=axes[0, 0],
                             vlim=(vmin, vmax), show=False)
axes[0, 0].set(title='CND1 f1')
cp.add_snr_colorbar(fig, axes[0, 0], im)
# Topography map of CND1 1f2:
im, _ = mne.viz.plot_topomap(topo_cnd2_1f1, epochs.info,
                             axes=axes[0, 1],
                             vlim=(vmin, vmax), show=False)
axes[0, 1].set(title='CND2 f1')
cp.add_snr_colorbar(fig, axes[0, 1], im)

# -------------------------
# @@@ f2 boost pairs

topo_cnd1_1f2 = np.array(df_pool['topo_cnd1_1f2'].tolist())
topo_cnd2_1f2 = np.array(df_pool['topo_cnd2_1f2'].tolist())

topo_cnd1_1f2 = np.mean(topo_cnd1_1f2 / np.max(topo_cnd1_1f2,
                                               axis=1, keepdims=True), axis=0)
topo_cnd2_1f2 = np.mean(topo_cnd2_1f2 / np.max(topo_cnd2_1f2,
                                               axis=1, keepdims=True), axis=0)

vmin = np.min([topo_cnd1_1f2, topo_cnd2_1f2])
vmax = np.max([topo_cnd1_1f2, topo_cnd2_1f2])

# Topography map of CND2 1f2:
im, _ = mne.viz.plot_topomap(topo_cnd2_1f2, epochs.info,
                             axes=axes[1, 0],
                             vlim=(vmin, vmax), show=False)
axes[1, 0].set(title='CND2 f2')
add_snr_colorbar(fig, axes[1, 0], im)
# Topography map of CND2, 1f1:
im, _ = mne.viz.plot_topomap(topo_cnd1_1f2, epochs.info,
                             axes=axes[1, 1],
                             vlim=(vmin, vmax), show=False)
axes[1, 1].set(title='CND1 f2')
add_snr_colorbar(fig, axes[1, 1], im)

# -------------------------
# @@@ ADD OBJECT BOOST MAPS

topo_boost_1f1 = np.array(df_pool['topo_boost_1f1'].tolist())
topo_boost_1f2 = np.array(df_pool['topo_boost_1f2'].tolist())

topo_boost_1f1 = np.mean(topo_boost_1f1 / np.max(topo_boost_1f1,
                                                 axis=1, keepdims=True),
                         axis=0)
topo_boost_1f2 = np.mean(topo_boost_1f2 / np.max(topo_boost_1f2,
                                                 axis=1, keepdims=True),
                         axis=0)

vmin = np.min([topo_boost_1f1, topo_boost_1f2])
vmax = np.max([topo_boost_1f1, topo_boost_1f2])

# Topography map of 1f1 boost:
im, _ = mne.viz.plot_topomap(topo_boost_1f1, epochs.info,
                             axes=axes[0, 2],
                             vlim=(vmin, vmax), show=False)
axes[0, 2].set(title=f'f1 boost ({f_label[0]})')
cp.add_snr_colorbar(fig, axes[0, 2], im)
# Topography map of 1f2 boost:
im, _ = mne.viz.plot_topomap(topo_boost_1f2, epochs.info,
                             axes=axes[1, 2],
                             vlim=(vmin, vmax), show=False)
axes[1, 2].set(title=f'f2 boost ({f_label[1]})')
cp.add_snr_colorbar(fig, axes[1, 2], im)

# save figure
plt.savefig(os.path.join(result_folder,
                         f'Pop_{task_show_names[task_number]}'
                         f'_Fig03_Topomap_cnds.pdf'))
# ---------------------------------------------------------------------

# # @@@ PLOT AVERAGE SNR CHANNELS @@@

fig, axs = plt.subplots(1, 3, figsize=(5, 4), sharey=False)

# ---------------------------------
# all channels

avg_boost_1f1_allCh = np.array(df_pool['avg_boost_1f1_allCh'].tolist())
avg_boost_1f2_allCh = np.array(df_pool['avg_boost_1f2_allCh'].tolist())

fig.suptitle(f'Pop. avg. SNR boost - {task_show_names[task_number]}'
             f'(N = {nsub})')

axs[0].bar([1, 2], [avg_boost_1f1_allCh.mean(), avg_boost_1f2_allCh.mean()],
           color='silver')
axs[0].plot(np.ones(nsub), avg_boost_1f1_allCh, 'o', mec='black', mfc='none')
axs[0].plot(2 * np.ones(nsub), avg_boost_1f2_allCh, 'o', mec='black',
            mfc='none')
axs[0].set(title='All channels', xticks=[1, 2],
           xticklabels=f_label, ylabel='SNR boost [%]')
cp.trim_axes(axs[0])
cp.prep4ai()

# ---------------------------------
# occipital channels

avg_boost_1f1_occCh = np.array(df_pool['avg_boost_1f1_occCh'].tolist())
avg_boost_1f2_occCh = np.array(df_pool['avg_boost_1f2_occCh'].tolist())

fig.suptitle(f'Pop. avg. SNR boost - {task_show_names[task_number]} '
             f'(N={nsub})')

axs[1].bar([1, 2], [avg_boost_1f1_occCh.mean(), avg_boost_1f2_occCh.mean()],
           color='silver')
axs[1].plot(np.ones(nsub), avg_boost_1f1_occCh, 'o', mec='black', mfc='none')
axs[1].plot(2 * np.ones(nsub), avg_boost_1f2_occCh, 'o', mec='black',
            mfc='none')
axs[1].set(title='Occ. channels', xticks=[1, 2],
           xticklabels=f_label)
cp.trim_axes(axs[1])
cp.prep4ai()

# ---------------------------------
# best channel

avg_boost_1f1_bestCh = np.array(df_pool['avg_boost_1f1_bestCh'].tolist())
avg_boost_1f2_bestCh = np.array(df_pool['avg_boost_1f2_bestCh'].tolist())

fig.suptitle(f'Pop. avg. SNR boost - {task_show_names[task_number]} '
             f'(N={nsub})')

axs[2].bar([1, 2], [avg_boost_1f1_bestCh.mean(), avg_boost_1f2_bestCh.mean()],
           color='silver')
axs[2].plot(np.ones(nsub), avg_boost_1f1_bestCh, 'o', mec='black', mfc='none')
axs[2].plot(2 * np.ones(nsub), avg_boost_1f2_bestCh, 'o', mec='black',
            mfc='none')
axs[2].set(title='Best occ. ch.', xticks=[1, 2],
           xticklabels=f_label)
cp.trim_axes(axs[2])
cp.prep4ai()

# ---------------------------------
# adjust and save figure
plt.tight_layout()
plt.savefig(os.path.join(result_folder,
                         f'Pop_{task_show_names[task_number]}'
                         f'_Fig04_SNR_boost.pdf'))
