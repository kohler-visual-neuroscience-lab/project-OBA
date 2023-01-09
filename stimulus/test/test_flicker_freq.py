"""
to test if the desired flickering frequency of an image is actually achived

Mohammad Shams
m.shams.ahmar@gmail.com
initiated on:       2022-11-08
last modified on:   2022-11-08
"""
from psychopy import event, visual, core
import lib as sup
import matplotlib.pyplot as plt
import numpy as np
import math
from egi_pynetstation.NetStation import NetStation

# #################################################
#                   INITIALIZE
# #################################################
# # Set an IP address for the computer running NetStation as an IPv4 string
# IP_ns = '10.10.10.42'
# # Set a port that NetStation will be listening to as an integer
# port_ns = 55513
# ns = NetStation(IP_ns, port_ns)
# # Set an NTP clock server (the amplifier) address as an IPv4 string
# IP_amp = '10.10.10.51'
# ns.connect(ntp_ip=IP_amp)

# configure the monitor and the stimulus window
mon = sup.config_mon_imac24()
win = sup.config_win(mon=mon, fullscr=False)

# measure the actual frame rate
actual_fr = win.getActualFrameRate(nIdentical=10,
                                   nMaxFrames=100,
                                   nWarmUpFrames=10,
                                   threshold=1)
print(f'measured refresh rate: {actual_fr}')
REF_RATE = 60

TRIAL_DUR = 300  # duration of a trial in [frames]

image_directory = "image/face.png"
IMAGE_FREQ = 7.5
IMAGE_DUR = REF_RATE / IMAGE_FREQ

# load image
image = visual.ImageStim(win, image=image_directory, pos=(0, 0))

# set a timer
timer = core.Clock()
# #################################################
#                 RUN THE TEST
# #################################################
# # Begin recording
# ns.begin_rec()
# # send a trigger to indicate the trial start
# ns.send_event(event_type="STRT", label='Start')

tt = np.full([math.ceil(IMAGE_FREQ*TRIAL_DUR/REF_RATE), 1], np.nan)
timer.reset()
show_counter = 0
image_curr_stat = False  # a flag to determine image appearance status

for iframe in range(TRIAL_DUR):
    sup.escape_session()  # allow force exit with 'escape' button
    # draw image on certain frames
    if sup.decide_on_show(iframe, IMAGE_DUR):
        image_prev_stat = image_curr_stat
        image_curr_stat = True
        image.draw()
        if image_curr_stat and not image_prev_stat:
            tt[show_counter] = timer.getTime()*1000
            time_old = tt[show_counter]
            show_counter = show_counter+1
            timer.reset()
    else:
        image_curr_stat = False

    win.flip()

actual_image_dur = np.median(tt)
actual_image_frq = 1000/actual_image_dur
print('==============================================')
print(f'Desired Flicker Frequency = {IMAGE_FREQ} Hz')
print(f'Median Flicker Frequency = {actual_image_frq} Hz')
print('==============================================')

tt = np.delete(tt, 0)
tt_freq = 1000/tt
bins = np.arange(IMAGE_FREQ-.05, IMAGE_FREQ+.05, .01)
plt.hist(tt_freq, bins=bins)
plt.ylabel('Count')
plt.xlabel('Frequency [Hz]')
plt.show()
# send a trigger to indicate the trial end
# ns.send_event(event_type="STOP", label='End')
# #################################################
#                   DISCONNECT
# #################################################
# # With the experiment concluded, you can end the recording
# ns.end_rec()
# # You'll want to disconnect the amplifier when your program is done
# ns.disconnect()
