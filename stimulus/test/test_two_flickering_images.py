"""
to test if the two frequencies tagged with the two image can be seen in XDIVA

Mohammad Shams
m.shams.ahmar@gmail.com
initiated on:       2022-11-08
last modified on:   2022-11-18
"""
from psychopy import visual
import lib as sup
import random
from egi_pynetstation.NetStation import NetStation

netstation = True  # decide whether to connect with NetStation

# #################################################
#                   INITIALIZE
# #################################################
if netstation:
    # Set an IP address for the computer running NetStation as an IPv4 string
    IP_ns = '10.10.10.42'
    # Set a port that NetStation will be listening to as an integer
    port_ns = 55513
    ns = NetStation(IP_ns, port_ns)
    # Set an NTP clock server (the amplifier) address as an IPv4 string
    IP_amp = '10.10.10.51'
    ns.connect(ntp_ip=IP_amp)

# configure the monitor and the stimulus window
mon = sup.config_mon_dell()
win = sup.config_win(mon=mon, fullscr=True)

# measure the actual frame rate
actual_fr = win.getActualFrameRate(nIdentical=10,
                                   nMaxFrames=100,
                                   nWarmUpFrames=10,
                                   threshold=1)
actual_fr = round(actual_fr, 2)
REF_RATE = 60
print(f"Nominal refresh rate: {REF_RATE} Hz")
print(f"Measured refresh rate: {actual_fr} Hz")

NTRIALS = 30  # number of trials
TRIAL_DUR = 600  # duration of a trial in [frames]
PAUSE_DUR = 120  # duration of the inter-trial pause
image1_directory = "image/house.png"
image2_directory = "image/face.png"

IMAGE1_DUR = REF_RATE / 7.5
IMAGE2_DUR = REF_RATE / 12

# #################################################
#                 RUN THE TEST
# #################################################
if netstation:
    # Begin recording
    ns.begin_rec()

try:
    for itrial in range(NTRIALS):
        print(f"starting trial {itrial}...")
        # load image and randomize their horizontal order
        image1_xpos = random.choice([+3.5, -3.5])
        image1 = visual.ImageStim(win, image=image1_directory,
                                  pos=(image1_xpos, 0))
        image2 = visual.ImageStim(win, image=image2_directory,
                                  pos=(-image1_xpos, 0))

        # determine which condition we are in
        if image1_xpos == 3.5:
            cnd = 1  # house on the right side
        else:
            cnd = 2  # house on the left side

        # run gap period
        for igap in range(PAUSE_DUR):
            win.flip()

        if netstation:
            # send a trigger to indicate the trial start
            ns.send_event(event_type="TRON", label=f'T{itrial}_begin_cnd{cnd}')

        for iframe in range(TRIAL_DUR):
            sup.escape_session()  # allow force exit with 'escape' button
            # add the fixation cross
            sup.draw_fixdot(win=win, size=.7, pos=(0, 0))
            # draw image on certain frames
            if sup.decide_on_show(iframe, IMAGE1_DUR):
                image1.draw()
            if sup.decide_on_show(iframe, IMAGE2_DUR):
                image2.draw()
            win.flip()

        if netstation:
            # send a trigger to indicate the trial end
            ns.send_event(event_type="TROF", label=f"T{itrial}_end_cnd{cnd}")
except:
    print(f"### Terminated the stimulus run during trial {itrial}\
    due to and error. ###")
    if netstation:
        ns.send_event(event_type="ERRS", label=f"stimulus_run_terminated")

finally:
    # #################################################
    #                   DISCONNECT
    # #################################################
    if netstation:
        # With the experiment concluded, you can end the recording
        ns.end_rec()
        # You'll want to disconnect the amplifier when your program is done
        # ns.disconnect()
