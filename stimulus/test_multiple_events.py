from psychopy import event, visual, core
import supplements as sup
import pandas as pd
import numpy as np

# configure the monitor and the stimulus window
mon = sup.config_mon_dell()
win = sup.config_win(mon=mon, fullscr=False, screen=0)

key_array = []
for iframe in range(600):
    key_response = event.getKeys(keyList=['space', 'escape'])
    if 'space' in key_response:
        print(key_response)
        key_array.append(key_response)
        event.clearEvents()
win.flip()
print(key_array)

# timer = core.Clock()
# timer.reset()
# a = np.empty((3,))
# a[:] = np.nan
# print(a)
# for i in range(3):
#     a[i] = timer.getTime()
# print(a)
