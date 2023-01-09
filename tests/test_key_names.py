from psychopy import event, visual, core
from lib import stim_flow_control as sup
import pandas as pd
import numpy as np

# configure the monitor and the stimulus window
mon = sup.config_mon_dell()
win = sup.config_win(mon=mon, fullscr=False, screen=0)

key_array = []
for iframe in range(100):
    win.flip()
    key_response = event.waitKeys()
    print(key_response)

