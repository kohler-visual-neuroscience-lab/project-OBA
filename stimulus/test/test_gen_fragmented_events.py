import gen_fragmented_events as frag
import matplotlib.pyplot as plt
import numpy as np

REF_RATE = 60
len_arr = np.empty((1000,))
times = []
for i in range(1000):
    event_frames = frag.gen_events(REF_RATE)
    len_arr[i] = len(event_frames)
    times = np.hstack((times, event_frames))

fig, axs = plt.subplots(2, 1)
axs[0].hist(len_arr, bins=list(range(0, 7)))
axs[0].set_xlabel('Number of events')
axs[0].set_ylabel('Count')
axs[1].hist(times, bins=list(range(0, 9 * REF_RATE, int(REF_RATE / 2))))
axs[1].set_xlabel('Time from trial onset (frames)')
axs[1].set_ylabel('Count')
fig.tight_layout()
plt.show()
