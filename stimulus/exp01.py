"""
Two superimposed images flicker near the fixation cross and subjects task is
to report a brief change in the contrast of the target image. Simultaneously
the same two images are shown on each side of the fixation cross and flicker
at two different frequencies

Mohammad Shams <m.shams.ahmar@gmail.com>
initiated on:       2022-11-25
last modified on:   2022-12-01
"""
import math
from psychopy import event, visual, core
import supplements as sup
import gen_random_path as gen_path
import pandas as pd
import numpy as np
import random
import os

# from egi_pynetstation.NetStation import NetStation

# -------------------------------------------------
# find out the last recorded block number
# -------------------------------------------------
temp_data = 'temp.json'
try:
    # read from file
    df = pd.read_json(temp_data)
    # update the block number
    iblock = df.last_block_num[0] + 1
    df.last_block_num[0] = iblock
    # write to file
    df.to_json(temp_data)
except:
    iblock = 1
    # create a dictionary of variables to be saved
    trial_dict = {'last_block_num': [iblock]}
    # convert to data frame
    df = pd.DataFrame(trial_dict)
    # write to file
    df.to_json(temp_data)
# -------------------------------------------------
# insert session meta data
# -------------------------------------------------
person = 'test'
session = '01'
N_BLOCKS = 1
N_TRIALS = 4  # must be an even number
screen_num = 0  # 0: primary    1: secondary
full_screen = False
netstation = False  # decide whether to connect with NetStation
# -------------------------------------------------
# destination file
# -------------------------------------------------
date = sup.get_date()
file_name = f"Exp01_{date}_{person}_S{session}.json"
data_path = os.path.join('Data/RawData', file_name)
# -------------------------------------------------
# initialize netstation at the beginning of the first block
# -------------------------------------------------
if netstation:
    # Set an IP address for the computer running NetStation as an IPv4 string
    IP_ns = '10.10.10.42'
    # Set a port that NetStation will be listening to as an integer
    port_ns = 55513
    ns = NetStation(IP_ns, port_ns)
    # Set an NTP clock server (the amplifier) address as an IPv4 string
    IP_amp = '10.10.10.51'
    ns.connect(ntp_ip=IP_amp)
    # Begin recording
    ns.begin_rec()
# -------------------------------------------------
# initialize the display
# -------------------------------------------------
TRIAL_DUR = 600  # duration of a trial in [frames]
ITI_DUR = 120  # inter-trial interval [frames]
REF_RATE = 60

# configure the monitor and the stimulus window
mon = sup.config_mon_dell()
win = sup.config_win(mon=mon, fullscr=full_screen, screen=screen_num)
sup.test_refresh_rate(win, REF_RATE)

# fixation cross
FIX_SIZE = .7
FIX_OFFSET = 5  # deg
FIX_X = 0
FIX_Y = 0

INSTRUCT_DUR = 30  # duration of the instruction period [frames]
# -------------------------------------------------
# set image properties and load
# -------------------------------------------------
image1_directory = "images/face_orient0.png"
image2_directory = "images/house_orient0.png"

# size [deg]
size_factor = 7
IMAGE1_SIZE = (size_factor, size_factor)
IMAGE2_SIZE = (size_factor, size_factor)
IMAGE3_SIZE = (size_factor, size_factor)

# opacity (1: opac | 0: transparent)
IMAGE_OPACITY = .4

# jittering properties
JITTER_REPETITION = 12  # number of frames where the relevant images keep
# their positions

REL_IMGPATH_N = TRIAL_DUR // JITTER_REPETITION + 1
REL_IMGPATH_SIGMA = .03
REL_IMGPATH_STEP = .03

REL_IMAGE_POS0_X = 0
REL_IMAGE_POS0_Y = 4

IRR_IMAGE_X = 4
IRR_IMAGE1_POS_Y = -2.5
IRR_IMAGE2_POS_Y = -2.5

# irr_image1_freq = random.choice([7.5, 12])
# IRR_IMAGE2_FREQ = np.setdiff1d(np.array([7.5, 12]), irr_image1_freq)
irr_image1_freq = 7.5
irr_image2_freq = 12

IRR_IMAGE1_nFRAMES = REF_RATE / irr_image1_freq
IRR_IMAGE2_nFRAMES = REF_RATE / irr_image2_freq

# possible frames, in which change can happen
change_frame_list = list(range(480, 540 + 1, 1))
# duration of changed-image [frames]
CHANGE_DUR = 18

# load images
rel_image1 = visual.ImageStim(win,
                              image=image1_directory,
                              size=IMAGE1_SIZE,
                              opacity=IMAGE_OPACITY)
rel_image2 = visual.ImageStim(win,
                              image=image2_directory,
                              size=IMAGE2_SIZE,
                              opacity=IMAGE_OPACITY)

# potential gap durations
gap_dur_list = range(30, 60 + 1, 1)

# probability of the valid cues
p_valid_cue = .5
# -------------------------------------------------
# define a timer to measure the change-detection reaction time
# -------------------------------------------------
timer = core.Clock()
change_time = float("nan")
response_time = float("nan")

irr_image1_pos_x = np.concatenate((np.repeat(IRR_IMAGE_X, N_TRIALS / 2),
                                   np.repeat(-IRR_IMAGE_X, N_TRIALS / 2)))
np.random.shuffle(irr_image1_pos_x)
irr_image2_pos_x = -irr_image1_pos_x

# show a message before the block begins
sup.block_msg(win, iblock)

# hide the cursor
mouse = event.Mouse(win=win, visible=False)
# calculate the first trial number of the current block
acc_trial = (iblock - 1) * N_TRIALS
# #################################################
#                   TRIAL itrial
# #################################################
for itrial in range(N_TRIALS):
    acc_trial += 1
    print(f"[Trial {acc_trial:02d}]   ", end="")
    # -------------------------------------------------
    # set up the stimulus behavior in current trial
    # -------------------------------------------------
    # find out in which order the images appear
    if irr_image2_pos_x[itrial] == IRR_IMAGE_X:
        order = 1  # Face - House
    else:
        order = 2  # House - Face

    # randomly decide on gap duration
    gap_dur = random.choice(gap_dur_list)

    # randomly decide on the time of the change
    change_frame = random.choice(change_frame_list)

    # randomly decide on which image to cue (show in the beginning)
    target_image = random.choice([1, 2])

    # define the condition number
    if (order == 1) & (target_image == 1):
        cnd = 1
    elif (order == 2) & (target_image == 1):
        cnd = 2
    elif (order == 1) & (target_image == 2):
        cnd = 3
    elif (order == 2) & (target_image == 2):
        cnd = 4
    else:
        cnd = None

    # on a proportion of trials change the non-target image
    if np.random.choice([True, False], p=[p_valid_cue, 1 - p_valid_cue]):
        change_image = target_image
    else:
        if target_image == 1:
            change_image = 2
        else:
            change_image = 1
    print(f"CueImg: {target_image}   TarImg: {change_image}   Cnd: {cnd}   ",
          end="")
    # ------------------------------------------------- setup end

    # load irrelevant images
    irr_image1 = visual.ImageStim(win,
                                  image=image1_directory,
                                  pos=(irr_image1_pos_x[itrial],
                                       IRR_IMAGE1_POS_Y),
                                  size=IMAGE1_SIZE,
                                  opacity=IMAGE_OPACITY)
    irr_image2 = visual.ImageStim(win,
                                  image=image2_directory,
                                  pos=(irr_image2_pos_x[itrial],
                                       IRR_IMAGE2_POS_Y),
                                  size=IMAGE2_SIZE,
                                  opacity=IMAGE_OPACITY)
    # generate the brownian path
    path1_x = gen_path.brownian_2d(
        n_samples=REL_IMGPATH_N,
        distribution_sigma=REL_IMGPATH_SIGMA,
        max_step=REL_IMGPATH_STEP) + REL_IMAGE_POS0_X
    path1_y = gen_path.brownian_2d(
        n_samples=REL_IMGPATH_N,
        distribution_sigma=REL_IMGPATH_SIGMA,
        max_step=REL_IMGPATH_STEP) + REL_IMAGE_POS0_Y

    path2_x = gen_path.brownian_2d(
        n_samples=REL_IMGPATH_N,
        distribution_sigma=REL_IMGPATH_SIGMA,
        max_step=REL_IMGPATH_STEP) + REL_IMAGE_POS0_X
    path2_y = gen_path.brownian_2d(
        n_samples=REL_IMGPATH_N,
        distribution_sigma=REL_IMGPATH_SIGMA,
        max_step=REL_IMGPATH_STEP) + REL_IMAGE_POS0_Y

    # slow down the jittering speed by reducing the position change rate
    path1_x = np.repeat(path1_x, JITTER_REPETITION)
    path1_y = np.repeat(path1_y, JITTER_REPETITION)
    path2_x = np.repeat(path2_x, JITTER_REPETITION)
    path2_y = np.repeat(path2_y, JITTER_REPETITION)

    # load the changed image
    if change_image == 1:
        image3_directory = "images/face_orient6.png"
    else:
        image3_directory = "images/house_orient6.png"

    rel_image3 = visual.ImageStim(win,
                                  image=image3_directory,
                                  size=IMAGE3_SIZE,
                                  opacity=IMAGE_OPACITY)
    # -------------------------------------------------
    # run the stimulus
    # -------------------------------------------------
    # set response state to no response
    response = 0  # 0: no change detected | 1: change detected
    # preassign response variables
    change_time = float("nan")
    response_time = float("nan")
    RT = float("nan")
    # reset the timer
    timer.reset()
    # ------------------
    # instruction period
    # ------------------
    # run gap period
    for igap in range(random.choice(gap_dur_list)):
        win.flip()
    # run the cue stimulus
    for iframe_instruction in range(INSTRUCT_DUR):
        if target_image == 1:
            rel_image1.pos = (0, 0)
            rel_image1.draw()
        else:
            rel_image2.pos = (0, 0)
            rel_image2.draw()
        win.flip()
    # run gap period
    for igap in range(random.choice(gap_dur_list)):
        win.flip()
    # ------------------
    # main period
    # ------------------
    # clear any response collected by getKey()
    event.clearEvents()
    if netstation:
        # send a trigger to indicate beginning of each trial
        ns.send_event(event_type=f"CND{cnd}",
                      label=f"CND{cnd}")
    for iframe in range(TRIAL_DUR):
        # flip frames as long as no response has been given
        if not response:
            sup.escape_session()  # force exit with 'escape' button
            # set the position of each task-relevant image
            rel_image1.pos = (path1_x[iframe], path1_y[iframe])
            rel_image2.pos = (path2_x[iframe], path2_y[iframe])

            # if conditions satisfied change the image
            if (iframe > change_frame) & (
                    iframe < change_frame + CHANGE_DUR):
                rel_image3.pos = (path2_x[iframe], path2_y[iframe])
                if change_image == 1:
                    rel_image2.draw()
                    rel_image3.draw()
                elif change_image == 2:
                    rel_image3.draw()
                    rel_image1.draw()
            # if not, show the unchanged versions
            else:
                rel_image2.draw()
                rel_image1.draw()

            # get the time of change
            if iframe == change_frame:
                change_time = timer.getTime()

            # draw irrelevant images conditionally
            if sup.decide_on_show(iframe, IRR_IMAGE1_nFRAMES):
                irr_image1.draw()
            if sup.decide_on_show(iframe, IRR_IMAGE2_nFRAMES):
                irr_image2.draw()
            sup.draw_fixdot(win=win, size=FIX_SIZE, pos=(FIX_X, FIX_Y))
            win.flip()

            # get response
            response_key = event.getKeys(keyList=['space'])
            if 'space' in response_key:
                response = 1
                response_time = timer.getTime()
                # measure reaction time in ms and end trial
                RT = round((response_time - change_time) * 1000, 0)
                if math.isnan(RT):
                    print(f"RT: Early")
                else:
                    print(f"RT: {RT} ms")
    # ------------------ main period ends
    if response == 0:
        print(f"RT: None")
    # run the inter-trial period
    for igap in range(ITI_DUR):
        win.flip()
    # -------------------------------------------------
    # create data frame and save
    # -------------------------------------------------
    # create a dictionary of variables to be saved
    trial_dict = {'trial_num': [acc_trial],
                  'condition_num': [cnd],
                  'image_order': [order],
                  'change_image': [change_image],
                  'target_image': [target_image],
                  'response_given': [response],
                  'response_time': [RT]}
    # convert to data frame
    dfnew = pd.DataFrame(trial_dict)
    # if first trial create a file, else load and add the new data frame
    if acc_trial == 1:
        dfnew.to_json(data_path)
    else:
        df = pd.read_json(data_path)
        dfnew = pd.concat([df, dfnew], ignore_index=True)
        dfnew.to_json(data_path)

    # clear the window buffer
    # win.depthMask = True
    # win.clearBuffer(color=False, depth=True, stencil=False)

if iblock == N_BLOCKS:
    # remove the temorary file
    os.remove(temp_data)
    # disconnect the amplifier
    if netstation:
        ns.disconnect()
    print(f"\n    *** Run finished and recording stopped ***")
    print('\n=======================================================')

else:
    if netstation:
        ns.end_rec()
    print(f"\n    Block {iblock} out of {N_BLOCKS} "
          f"finished and recording paused...")
    print('\n=======================================================')

win.close()
