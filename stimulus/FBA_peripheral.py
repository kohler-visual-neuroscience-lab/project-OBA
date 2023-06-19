"""
***** Object-based attention (OBA) project
***** Experiment 01: peripheral flickering features

    Mo Shams <MShamsCBR@gmail.com>
    June 01, 2023

This is a modified version of pilot02.py

Two superimposed images (relevant) appear while the subject fixates at the
center of the them. One of the two images is cued in the beginning of each
trial and subject is prompted to detect a tilt (zero to two times in each
trial) by pressing a key.
Simultaneously, a ring of eight copies of the same images (irrelevant)
flicker at their corresponding frequencies (either 7.5Hz or 12Hz).

There are two conditions:
    CND1: attend blue
    CND2: attend red


"""
import os
import random
import numpy as np
import pandas as pd
from lib import stim_flow_control as sfc, gen_events, \
    gen_random_path as gen_path
from psychopy import event, visual, core
from lib.evaluate_responses import eval_resp
from egi_pynetstation.NetStation import NetStation


def pol2cart(rho, phi):
    x_cart = rho * np.cos(phi)
    y_cart = rho * np.sin(phi)
    return x_cart, y_cart


# disable Panda's false warning message
pd.options.mode.chained_assignment = None  # default='warn'

# ----------------------------------------------------------------------------
# /// INSERT SESSION'S META DATA ///

subID = "5015"
N_BLOCKS = 2  # (2)
N_TRIALS = 32  # (32) number of trials per block (must be a factor of TWO)
screen_num = 1  # 0: ctrl room    1: test room
full_screen = True  # (True/False)
netstation = True  # (True/False) decide whether to connect with NetStation
keyboard = "numpad"  # numpad/mac
freq1 = 7.5
freq2 = 12
# ----------------------------------------------------------------------------

# /// CONFIGURE LOAD/SAVE FILES & DIRECTORIES ///

# find out the last recorded block number
temp_data = 'temp.json'
try:
    # read from file
    df = pd.read_json(temp_data)
    # read file name
    file_name = df.file_name[0]
    # update the block number
    iblock = df['last_block_num'][0] + 1
    df['last_block_num'][0] = iblock
    # write to file
    df.to_json(temp_data)
except:
    iblock = 1
    # create file name
    date = sfc.get_date()
    time = sfc.get_time()
    file_name = f"{subID}_{date}_{time}_exp01_fba_per.json"
    # create a dictionary of variables to be saved
    trial_dict = {'last_block_num': [iblock],
                  'file_name': [file_name]}
    # convert to data frame
    df = pd.DataFrame(trial_dict)
    # Note: saving is postponed to the end of the first trial

# set data directory
data_path = os.path.join("..", "data", "raw", file_name)
# set image root directory
image_root = os.path.join("image", "image_set_exp01")
# ----------------------------------------------------------------------------

# /// CONFIGURE ECI CONNECTION ///

# initialize netstation at the beginning of the first block
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
else:
    ns = None

# ----------------------------------------------------------------------------

# /// CONFIGURE STIMULUS PARAMETERS AND INPUTS ///

# initialize the display and the keyboard
REF_RATE = 60
TRIAL_DUR = 7 * REF_RATE  # duration of a trial [frames]
ITI_DUR = 2 * REF_RATE  # inter-trial interval [frames]
N_CNDS = 2

# configure the monitor and the stimulus window
mon = sfc.config_mon_dell()
win = sfc.config_win(mon=mon, fullscr=full_screen, screen=screen_num)
sfc.test_refresh_rate(win, REF_RATE)

# fixation cross
FIX_SIZE = .5
FIX_X = 0
FIX_Y = 0

INSTRUCT_DUR = REF_RATE  # duration of the instruction period [frames]

if keyboard == "numpad":
    command_keys = {"quit_key": "backspace", "response_key": "num_insert"}
elif keyboard == "mac":
    command_keys = {"quit_key": "escape", "response_key": "space"}
else:
    raise NameError(f"Keyboard name '{keyboard}' not recognized.")

# duration of changed-image [frames]
TILT_DUR = int(REF_RATE / 4)  # equal to 250 ms

# size [deg]
size_factor = 2.2
rel_image_size = np.array([size_factor, size_factor])
irr_image_size = np.array([size_factor, size_factor])

# opacity (1: opac | 0: transparent)
IMAGE_OPACITY_REL_FRONT = .5
IMAGE_OPACITY_REL_BACK = .7

# jittering properties
JITTER_REPETITION = int(REF_RATE / 20)  # number of frames where the relevant
# images keep their positions (50 ms; 20 Hz)

REL_IMGPATH_N = TRIAL_DUR // JITTER_REPETITION + 1
REL_IMGPATH_SIGMA = .01
REL_IMGPATH_STEP = .02

REL_IMAGE_POS0_X = FIX_X
REL_IMAGE_POS0_Y = FIX_Y

IRR_IMAGE_RHO = 3
IRR_IMAGE_THETA_deg = np.linspace(22.5, 360 + 22.5, 9)
IRR_IMAGE_THETA_deg = np.delete(IRR_IMAGE_THETA_deg, -1)
IRR_IMAGE_THETA = IRR_IMAGE_THETA_deg * (2 * np.pi) / 360
IRR_IMAGE_X, IRR_IMAGE_Y = pol2cart(IRR_IMAGE_RHO, IRR_IMAGE_THETA)

IRR_IMAGE_RHO2 = 5.5
IRR_IMAGE_THETA_deg = np.linspace(-22.5, 360-22.5, 9)
IRR_IMAGE_THETA_deg = np.delete(IRR_IMAGE_THETA_deg, -1)
IRR_IMAGE_THETA = IRR_IMAGE_THETA_deg * (2 * np.pi) / 360
IRR_IMAGE_X2, IRR_IMAGE_Y2 = pol2cart(IRR_IMAGE_RHO2, IRR_IMAGE_THETA)

# create two orders of appearance
# (no neighbours from same category allowed)
order1_x = IRR_IMAGE_X
order1_y = IRR_IMAGE_Y
order2_x = np.roll(IRR_IMAGE_X, 1)
order2_y = np.roll(IRR_IMAGE_Y, 1)

order1_x2 = IRR_IMAGE_X2
order1_y2 = IRR_IMAGE_Y2
order2_x2 = np.roll(IRR_IMAGE_X2, 1)
order2_y2 = np.roll(IRR_IMAGE_Y2, 1)

freq1 = 7.5
freq2 = 12

# potential gap durations (0.5 to 1 sec)
gap_dur_list = range(int(REF_RATE / 2), REF_RATE + 1, 1)

# define a timer to measure the change-detection reaction time
timer = core.Clock()

# show a message before the block begins
sfc.block_msg(win, iblock, N_BLOCKS, command_keys)

# hide the cursor
mouse = event.Mouse(win=win, visible=False)
# calculate the first trial number of the current block
acc_trial = (iblock - 1) * N_TRIALS

# create an equal number of trials per condition in current block
n_trials_per_cnd = int(N_TRIALS / N_CNDS)
cnd_array = np.hstack([np.ones(n_trials_per_cnd, dtype=int) * 1,
                       np.ones(n_trials_per_cnd, dtype=int) * 2])
np.random.shuffle(cnd_array)

# ----------------------------------------------------------------------------

# /// TRIAL BEGINS ///

for itrial in range(N_TRIALS):
    iblue = 1
    ired = 1

    # /// set up the stimulus behavior in current trial

    acc_trial += 1
    print(f"[Trial {acc_trial:03d}]   ", end="")
    if acc_trial > 1:
        # read current running performance
        df_temp = pd.read_json(data_path)
        prev_run_perf = df_temp.loc[acc_trial - 2, 'running_performance']
        prev_tilt_mag = df_temp.loc[acc_trial - 2, 'tilt_magnitude']
    else:
        prev_run_perf = None
        prev_tilt_mag = None

    # randomly select frames, in which change happens
    change_start_frames = gen_events.gen_events2(REF_RATE)
    n_total_evnts = len(change_start_frames)
    change_frames = np.array(change_start_frames)
    change_times = np.empty((n_total_evnts,))
    change_times[:] = np.nan
    response_times = [np.nan]

    for i in change_start_frames:
        for j in range(TILT_DUR - 1):
            change_frames = \
                np.hstack((change_frames, [i + j + 1]))

    # extract current trial's condition
    cnd = cnd_array[itrial - 1]
    print(f"Cnd: {cnd}   #Events: {n_total_evnts}   ", end="")

    # randomly decide on gap duration
    gap_dur = random.choice(gap_dur_list)

    # decide on the cue/target image
    if cnd == 1:
        cue_image = 1
    elif cnd == 2:
        cue_image = 2
    else:
        cue_image = None
        print("Invalid condition number!")

    IRR_IMAGE1_nFRAMES = REF_RATE / freq1
    IRR_IMAGE2_nFRAMES = REF_RATE / freq2

    # pick the tilting image for each event , independently of the cued image
    tilt_images = np.random.choice([1, 2], n_total_evnts)
    # pick the tilting direction for each event
    tilt_dirs = np.random.choice(['CW', 'CCW'], n_total_evnts)

    # --------------------------------
    # set image properties and load
    image1_directory = os.path.join(image_root, f"blue{iblue}_tilt0.png")
    image2_directory = os.path.join(image_root, f"red{ired}_tilt0.png")
    # assign transparency
    if cue_image == 1:
        image1_trans = IMAGE_OPACITY_REL_FRONT
        image2_trans = IMAGE_OPACITY_REL_BACK
    else:
        image2_trans = IMAGE_OPACITY_REL_FRONT
        image1_trans = IMAGE_OPACITY_REL_BACK
    # load relevant images
    rel_image1 = visual.ImageStim(win,
                                  image=image1_directory,
                                  size=rel_image_size,
                                  opacity=image1_trans)
    rel_image2 = visual.ImageStim(win,
                                  image=image2_directory,
                                  size=rel_image_size,
                                  opacity=image2_trans)

    # load irrelevant images (ring 1)
    irr_image11 = visual.ImageStim(win,
                                   image=image1_directory,
                                   size=irr_image_size)
    irr_image12 = visual.ImageStim(win,
                                   image=image1_directory,
                                   size=irr_image_size)
    irr_image13 = visual.ImageStim(win,
                                   image=image1_directory,
                                   size=irr_image_size)
    irr_image14 = visual.ImageStim(win,
                                   image=image1_directory,
                                   size=irr_image_size)
    irr_image21 = visual.ImageStim(win,
                                   image=image2_directory,
                                   size=irr_image_size)
    irr_image22 = visual.ImageStim(win,
                                   image=image2_directory,
                                   size=irr_image_size)
    irr_image23 = visual.ImageStim(win,
                                   image=image2_directory,
                                   size=irr_image_size)
    irr_image24 = visual.ImageStim(win,
                                   image=image2_directory,
                                   size=irr_image_size)

    # load irrelevant images (ring 2)
    irr_image31 = visual.ImageStim(win,
                                   image=image1_directory,
                                   size=irr_image_size)
    irr_image32 = visual.ImageStim(win,
                                   image=image1_directory,
                                   size=irr_image_size)
    irr_image33 = visual.ImageStim(win,
                                   image=image1_directory,
                                   size=irr_image_size)
    irr_image34 = visual.ImageStim(win,
                                   image=image1_directory,
                                   size=irr_image_size)
    irr_image41 = visual.ImageStim(win,
                                   image=image2_directory,
                                   size=irr_image_size)
    irr_image42 = visual.ImageStim(win,
                                   image=image2_directory,
                                   size=irr_image_size)
    irr_image43 = visual.ImageStim(win,
                                   image=image2_directory,
                                   size=irr_image_size)
    irr_image44 = visual.ImageStim(win,
                                   image=image2_directory,
                                   size=irr_image_size)

    # decide on the order of appearance
    order = np.random.choice([1, 2])
    if order == 1:
        irr_poss_x = order1_x
        irr_poss_y = order1_y
        irr_poss_x2 = order1_x2
        irr_poss_y2 = order1_y2
    else:
        irr_poss_x = order2_x
        irr_poss_y = order2_y
        irr_poss_x2 = order2_x2
        irr_poss_y2 = order2_y2
    # assign calculated postions to irrelevant images
    all_locs = [[irr_poss_x[0], irr_poss_y[0]],
                [irr_poss_x[1], irr_poss_y[1]],
                [irr_poss_x[2], irr_poss_y[2]],
                [irr_poss_x[3], irr_poss_y[3]],
                [irr_poss_x[4], irr_poss_y[4]],
                [irr_poss_x[5], irr_poss_y[5]],
                [irr_poss_x[6], irr_poss_y[6]],
                [irr_poss_x[7], irr_poss_y[7]],
                [irr_poss_x2[0], irr_poss_y2[0]],
                [irr_poss_x2[1], irr_poss_y2[1]],
                [irr_poss_x2[2], irr_poss_y2[2]],
                [irr_poss_x2[3], irr_poss_y2[3]],
                [irr_poss_x2[4], irr_poss_y2[4]],
                [irr_poss_x2[5], irr_poss_y2[5]],
                [irr_poss_x2[6], irr_poss_y2[6]],
                [irr_poss_x2[7], irr_poss_y2[7]]]

    # np.random.shuffle(all_locs)

    irr_image11.pos = all_locs[0]
    irr_image21.pos = all_locs[1]
    irr_image12.pos = all_locs[2]
    irr_image22.pos = all_locs[3]
    irr_image13.pos = all_locs[4]
    irr_image23.pos = all_locs[5]
    irr_image14.pos = all_locs[6]
    irr_image24.pos = all_locs[7]

    irr_image31.pos = all_locs[8]
    irr_image41.pos = all_locs[9]
    irr_image32.pos = all_locs[10]
    irr_image42.pos = all_locs[11]
    irr_image33.pos = all_locs[12]
    irr_image43.pos = all_locs[13]
    irr_image34.pos = all_locs[14]
    irr_image44.pos = all_locs[15]

    # --------------------------------

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

    if acc_trial == 1:
        tilt_mag = 30
        tilt_change = 0
    else:
        # calculate what titl angle (magnitude) to use
        tilt_change = sfc.cal_next_tilt(goal_perf=80, run_perf=prev_run_perf)
        tilt_mag = int(prev_tilt_mag + tilt_change)
        # take care of saturated scenarios
        if tilt_mag > 99:
            tilt_mag = 99
        elif tilt_mag < 1:
            tilt_mag = 1
    print(f"TiltAng: {(tilt_mag / 10):3.1f}deg   ", end="")

    # load the changed image
    image3_directory1cw = os.path.join(image_root,
                                       f"blue{iblue}_tilt{tilt_mag}_CW.png")
    image3_directory1ccw = os.path.join(image_root,
                                        f"blue{iblue}_tilt{tilt_mag}_CCW.png")
    image3_directory2cw = os.path.join(image_root,
                                       f"red{ired}_tilt{tilt_mag}_CW.png")
    image3_directory2ccw = os.path.join(image_root,
                                        f"red{ired}_tilt{tilt_mag}_CCW.png")

    rel_image3_1cw = visual.ImageStim(win,
                                      image=image3_directory1cw,
                                      size=rel_image_size,
                                      opacity=image1_trans)
    rel_image3_1ccw = visual.ImageStim(win,
                                       image=image3_directory1ccw,
                                       size=rel_image_size,
                                       opacity=image1_trans)
    rel_image3_2cw = visual.ImageStim(win,
                                      image=image3_directory2cw,
                                      size=rel_image_size,
                                      opacity=image2_trans)
    rel_image3_2ccw = visual.ImageStim(win,
                                       image=image3_directory2ccw,
                                       size=rel_image_size,
                                       opacity=image2_trans)
    # --------------------------------

    # /// run the stimulus

    cur_evnt_n = 0

    # gap period
    for igap in range(random.choice(gap_dur_list)):
        win.flip()

    # cue period
    cue_yoffset = 0
    for iframe_instruction in range(INSTRUCT_DUR):
        if cue_image == 1:
            rel_image1.pos = (0, cue_yoffset)
            rel_image1.draw()
        else:
            rel_image2.pos = (0, cue_yoffset)
            rel_image2.draw()
        win.flip()

    # run gap period
    for igap in range(random.choice(gap_dur_list)):
        sfc.draw_fixdot(win=win, size=FIX_SIZE,
                        pos=(FIX_X, FIX_Y),
                        cue=cue_image)
        win.flip()

    # tilt detection period
    timer.reset()
    if netstation:
        # send a trigger to indicate beginning of each trial
        ns.send_event(event_type=f"CND{cnd}",
                      label=f"CND{cnd}")
    for iframe in range(TRIAL_DUR):
        pressed_key = event.getKeys(keyList=list(command_keys.values()))
        # set the position of each task-relevant image
        rel_image1.pos = (path1_x[iframe], path1_y[iframe])
        rel_image2.pos = (path2_x[iframe], path2_y[iframe])

        # get the time of change
        if iframe in change_start_frames:
            ch_t = timer.getTime()
            change_times[cur_evnt_n] = round(ch_t * 1000)
            cur_evnt_n += 1

        # make sure the cued image stays on top
        if cue_image == 1:
            # if conditions satisfied tilt the image
            if iframe in change_frames:
                if tilt_dirs[cur_evnt_n - 1] == 'CW':
                    if tilt_images[cur_evnt_n - 1] == 1:
                        rel_image3_1cw.pos = (
                            path1_x[iframe], path1_y[iframe])
                        rel_image2.draw()
                        rel_image3_1cw.draw()
                    elif tilt_images[cur_evnt_n - 1] == 2:
                        rel_image3_2cw.pos = (
                            path2_x[iframe], path2_y[iframe])
                        rel_image3_2cw.draw()
                        rel_image1.draw()
                else:
                    if tilt_images[cur_evnt_n - 1] == 1:
                        rel_image3_1ccw.pos = (
                            path1_x[iframe], path1_y[iframe])
                        rel_image2.draw()
                        rel_image3_1ccw.draw()
                    elif tilt_images[cur_evnt_n - 1] == 2:
                        rel_image3_2ccw.pos = (
                            path2_x[iframe], path2_y[iframe])
                        rel_image3_2ccw.draw()
                        rel_image1.draw()
            # if not, show the unchanged versions
            else:
                rel_image2.draw()
                rel_image1.draw()

        elif cue_image == 2:
            if iframe in change_frames:
                if tilt_dirs[cur_evnt_n - 1] == 'CW':
                    if tilt_images[cur_evnt_n - 1] == 1:
                        rel_image3_1cw.pos = (
                            path1_x[iframe], path1_y[iframe])
                        rel_image3_1cw.draw()
                        rel_image2.draw()
                    elif tilt_images[cur_evnt_n - 1] == 2:
                        rel_image3_2cw.pos = (
                            path2_x[iframe], path2_y[iframe])
                        rel_image1.draw()
                        rel_image3_2cw.draw()
                else:
                    if tilt_images[cur_evnt_n - 1] == 1:
                        rel_image3_1ccw.pos = (
                            path1_x[iframe], path1_y[iframe])
                        rel_image3_1ccw.draw()
                        rel_image2.draw()
                    elif tilt_images[cur_evnt_n - 1] == 2:
                        rel_image3_2ccw.pos = (
                            path2_x[iframe], path2_y[iframe])
                        rel_image1.draw()
                        rel_image3_2ccw.draw()
            # if not, show the unchanged versions
            else:
                rel_image1.draw()
                rel_image2.draw()

        # draw irrelevant image conditionally
        if sfc.decide_on_show(iframe, IRR_IMAGE1_nFRAMES):
            irr_image11.draw()
            irr_image12.draw()
            irr_image13.draw()
            irr_image14.draw()

            irr_image31.draw()
            irr_image32.draw()
            irr_image33.draw()
            irr_image34.draw()

        if sfc.decide_on_show(iframe, IRR_IMAGE2_nFRAMES):
            irr_image21.draw()
            irr_image22.draw()
            irr_image23.draw()
            irr_image24.draw()

            irr_image41.draw()
            irr_image42.draw()
            irr_image43.draw()
            irr_image44.draw()

        sfc.draw_fixdot(win=win, size=FIX_SIZE,
                        pos=(FIX_X, FIX_Y),
                        cue=cue_image)
        win.flip()

        # response period
        if command_keys['quit_key'] in pressed_key:
            core.quit()
        # check if space bar is pressed within 1 sec from tilt
        if command_keys['response_key'] in pressed_key:
            res_t = timer.getTime()
            response_times.append(round(res_t * 1000))
    response_times.pop(0)
    # evaluate the response
    [instant_perf, avg_rt] = eval_resp(cue_image,
                                       tilt_images,
                                       change_times,
                                       response_times)
    if np.isnan(avg_rt):
        print(f"Perf:{int(instant_perf):3d}%   avgRT:  nan    ", end="")
    else:
        print(f"Perf:{int(instant_perf):3d}%   avgRT:{int(avg_rt):4d}ms   ",
              end="")

    # --------------------------------

    # /// prepare data for saving

    # create a dictionary of variables to be saved
    trial_dict = {'trial_num': [acc_trial],
                  'block_num': [iblock],
                  'condition_num': [cnd],
                  'Frequency_tags': [[freq1, freq2]],
                  'cued_image': [cue_image],
                  'image_order': [order],
                  'n_events': n_total_evnts,
                  'tilted_images': [tilt_images],
                  'tilt_directions': [tilt_dirs],
                  'tilt_magnitude': [tilt_mag],
                  'avg_rt': [avg_rt],
                  'instant_performance': [instant_perf],
                  'cummulative_performance': [np.nan],
                  'running_performance': [np.nan]}
    # convert to data frame
    dfnew = pd.DataFrame(trial_dict)
    # if not first trial, load the existing data frame and concatenate
    if acc_trial == 1:
        df.to_json(temp_data)  # to keep a record of the block number
    else:
        df = pd.read_json(data_path)
        dfnew = pd.concat([df, dfnew], ignore_index=True)

    # --------------------------------

    # /// calculate cummulative and running performances

    # calculate the cumulative performance (all recorded trials)
    eval_series = dfnew.instant_performance
    eval_array = eval_series.values
    cum_perf = round(sum(eval_array) / len(eval_array), 2)
    print(f"CumPerf:{cum_perf:6.2f}%   ", end="")
    # calculate the running performance (last 10 trials)
    run_perf = round(sum(eval_array[-10:]) / len(eval_array[-10:]), 2)
    print(f"RunPerf:{run_perf:6.2f}%")
    # fill the remaining values in the data frame
    dfnew.loc[acc_trial - 1,
    ['cummulative_performance',
     'running_performance']] = [cum_perf, run_perf]
    # save the data frame
    dfnew.to_json(data_path)

    # gap period (0.5 sec)
    for igap in np.arange(REF_RATE / 2):
        win.flip()

# -----------------------------------------------------------------------------

# /// STOP/PAUSE ECI ///

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
