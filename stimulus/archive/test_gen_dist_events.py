import gen_distributed_events as gen_evnts
import matplotlib.pyplot as plt
import numpy as np

REF_RATE = 60
TRIAL_DUR = 10 * REF_RATE
len_arr = np.empty((1000,))
times = []
for i in range(1000):
    change_start_frames = gen_evnts.gen_ditributed_event(0,
                                                         TRIAL_DUR,
                                                         REF_RATE,
                                                         TRIAL_DUR - REF_RATE,
                                                         REF_RATE * 2)
    len_arr[i] = len(change_start_frames)
    times = np.hstack((times, change_start_frames))

fig, ax = plt.subplots(2, 1)
ax[0].hist(len_arr, bins=list(range(0, 7, 1)))
ax[1] = plt.hist(times)
plt.show()
