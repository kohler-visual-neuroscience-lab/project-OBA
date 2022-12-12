import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = '0006_20221202_052011.mff'
beh_path = 'Exp01_20221202_0006_S01.json'
raw = mne.io.read_raw_egi(data_path, preload=True)
raw.info['line_freq'] = 60.
df = pd.read_json(beh_path)
# Set montage
montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
raw.set_montage(montage, match_alias=True)
# Set common average reference
raw.set_eeg_reference('average', projection=False, verbose=False)
# Apply bandpass filter
raw.filter(l_freq=0.1, h_freq=None, fir_design='firwin', verbose=False)
events = mne.find_events(raw, stim_channel='TRON')
events[:, 2] = df['condition_num']
event_id = {'cnd1': 1, 'cnd2': 2, 'cnd3': 3, 'cnd4': 4}
# times are wrt event times
tmin = 0  # in sec
tmax = 8  # in sec
epochs = mne.Epochs(raw,
                    events=events,
                    event_id=[event_id['cnd1'], event_id['cnd2'],
                              event_id['cnd3'], event_id['cnd4']],
                    tmin=tmin,
                    tmax=tmax,
                    baseline=None,
                    verbose=False)


freq = '2'

ID = '0006'
df = pd.read_json(f'rmsf{freq}_{ID}.json')
temp = df['cnd31_rms']
cnd31_1 = temp.to_numpy()
cnd31_1 = cnd31_1 / np.max(np.abs(cnd31_1))
temp = df['cnd24_rms']
cnd24_1 = temp.to_numpy()
cnd24_1 = cnd24_1 / np.max(np.abs(cnd24_1))

ID = '0007'
df = pd.read_json(f'rmsf{freq}_{ID}.json')
cnd31_2 = temp.to_numpy()
cnd31_2 = cnd31_2 / np.max(np.abs(cnd31_2))
temp = df['cnd24_rms']
cnd24_2 = temp.to_numpy()
cnd24_2 = cnd24_2 / np.max(np.abs(cnd24_2))

ID = '0009'
df = pd.read_json(f'rmsf{freq}_{ID}.json')
cnd31_3 = temp.to_numpy()
cnd31_3 = cnd31_3 / np.max(np.abs(cnd31_3))
temp = df['cnd24_rms']
cnd24_3 = temp.to_numpy()
cnd24_3 = cnd24_3 / np.max(np.abs(cnd24_3))

cnd31 = np.mean(np.vstack((cnd31_1, cnd31_2, cnd31_3)), axis=0)
cnd24 = np.mean(np.vstack((cnd24_1, cnd24_2, cnd24_3)), axis=0)

vlim1_31 = np.min(cnd31)
vlim2_31 = np.max(cnd31)
vlim1_24 = np.min(cnd24)
vlim2_24 = np.max(cnd24)

fig, ax = plt.subplots(1, 2)
fig.suptitle(f"Pop. mean")
ax[0].set_title(f'House_boost_f{freq}')
ax[1].set_title(f'Face_boost_f{freq}')
mne.viz.plot_topomap(cnd31, epochs.info, axes=ax[0], show=False,
                     vlim=(vlim1_31, vlim2_31))
mne.viz.plot_topomap(cnd24, epochs.info, axes=ax[1], show=False,
                     vlim=(vlim1_24, vlim2_24))
plt.savefig(f'topomap_diff_f{freq}_rms3harm_pop.png')
