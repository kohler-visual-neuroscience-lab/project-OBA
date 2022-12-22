"""
***** Object-based attention (OBA) project
***** Experiment 01

    Mohammad Shams <m.shams.ahmar@gmail.com>
    Initiated on:       2022-11-25

Two superimposed images flicker near the fixation cross and subject's task is
to report a brief tilt in the one of the images. Simultaneously, a copy of
the two images appear on each side of the fixation cross and flicker
at two different frequencies.

There are four conditions:
    cnd 1: face left, house right, attend face
    cnd 2: house left, face right, attend face
    cnd 3: face left, house right, attend house
    cnd 4: face left, house right, attend house

"""
import math
from psychopy import event, visual, core
import lib as sup
import gen_random_path as gen_path
import pandas as pd
import numpy as np
import random
import os
import gen_events
from egi_pynetstation.NetStation import NetStation

# disable the false-positive chained assignment warning
pd.options.mode.chained_assignment = None  # default='warn'

# -------------------------------------------------
# insert session meta data
# -------------------------------------------------
subID = 'test'
N_BLOCKS = 2
N_TRIALS = 4  # must be a factor of FOUR
screen_num = 1  # 0: primary    1: secondary
full_screen = True
netstation = True  # decide whether to connect with NetStation
# -------------------------------------------------
# find out the last recorded block number
# -------------------------------------------------
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
    date = sup.get_date()
    time = sup.get_time()
    file_name = f"beh_{date}_{time}_{subID}.json"
    # create a dictionary of variables to be saved
    trial_dict = {'last_block_num': [iblock],
                  'file_name': [file_name]}
    # convert to data frame
    df = pd.DataFrame(trial_dict)
    # write to file
    df.to_json(temp_data)
# -------------------------------------------------
# destination file
# -------------------------------------------------
data_path = os.path.join('data', file_name)
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
else:
    ns = None
# -------------------------------------------------
# initialize the display and the keyboard
# -------------------------------------------------
REF_RATE = 60
TRIAL_DUR = 10 * REF_RATE  # duration of a trial in [frames]
ITI_DUR = 2 * REF_RATE  # inter-trial interval [frames]

# configure the monitor and the stimulus window
mon = sup.config_mon_dell()
win = sup.config_win(mon=mon, fullscr=full_screen, screen=screen_num)
sup.test_refresh_rate(win, REF_RATE)

# fixation cross
FIX_SIZE = .7
FIX_OFFSET = 5  # deg
FIX_X = 0
FIX_Y = 0

INSTRUCT_DUR = REF_RATE  # duration of the instruction period [frames]

command_keys = {"quit_key": "backspace", "response": "num_insert"}
# -------------------------------------------------
# set image properties and load
# -------------------------------------------------
image1_directory = os.path.join("images", "face_tilt0.png")
image2_directory = os.path.join("images", "house_tilt0.png")

# size [deg]
size_factor = 7
IMAGE1_SIZE = (size_factor, size_factor)
IMAGE2_SIZE = (size_factor, size_factor)
IMAGE3_SIZE = (size_factor, size_factor)

# opacity (1: opac | 0: transparent)
IMAGE_OPACITY = .4

# jittering properties
JITTER_REPETITION = int(REF_RATE / 10)  # number of frames where the relevant
# images keep their positions

REL_IMGPATH_N = TRIAL_DUR // JITTER_REPETITION + 1
REL_IMGPATH_SIGMA = .7
REL_IMGPATH_STEP = .1

REL_IMAGE_POS0_X = 0
REL_IMAGE_POS0_Y = 4.5

IRR_IMAGE_X = 4
IRR_IMAGE1_POS_Y = -2.5
IRR_IMAGE2_POS_Y = -2.5

# irr_image1_freq = random.choice([7.5, 12])
# IRR_IMAGE2_FREQ = np.setdiff1d(np.array([7.5, 12]), irr_image1_freq)
irr_image1_freq = 7.5
irr_image2_freq = 12

IRR_IMAGE1_nFRAMES = REF_RATE / irr_image1_freq
IRR_IMAGE2_nFRAMES = REF_RATE / irr_image2_freq

# duration of changed-image [frames]
TILT_DUR = int(REF_RATE / 4)

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
gap_dur_list = range(int(REF_RATE / 2), REF_RATE + 1, 1)

# probability of the valid cues
p_valid_cue = .5
# -------------------------------------------------
# define a timer to measure the change-detection reaction time
# -------------------------------------------------
timer = core.Clock()

irr_image1_pos_x = np.concatenate((np.repeat(IRR_IMAGE_X, N_TRIALS / 2),
                                   np.repeat(-IRR_IMAGE_X, N_TRIALS / 2)))
np.random.shuffle(irr_image1_pos_x)
irr_image2_pos_x = -irr_image1_pos_x

# show a message before the block begins
sup.block_msg(win, iblock, command_keys)

# hide the cursor
mouse = event.Mouse(win=win, visible=False)
# calculate the first trial number of the current block
acc_trial = (iblock - 1) * N_TRIALS

# create an equal number of trials per condition in current block
n_trials_per_cnd = int(N_TRIALS / 4)
cnd_array = np.hstack([np.ones(n_trials_per_cnd, dtype=int),
                       np.ones(n_trials_per_cnd, dtype=int) * 2,
                       np.ones(n_trials_per_cnd, dtype=int) * 3,
                       np.ones(n_trials_per_cnd, dtype=int) * 4])
np.random.shuffle(cnd_array)
# #################################################
#                   TRIAL itrial
# #################################################
for itrial in range(N_TRIALS):
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
    change_start_frames = gen_events.gen_events(REF_RATE)
    n_total_evnts = len(change_start_frames)
    change_frames = np.array(change_start_frames)
    change_times = np.empty((n_total_evnts,))
    change_times[:] = np.nan
    response_times = [np.nan]

    for i in change_start_frames:
        for j in range(TILT_DUR - 1):
            change_frames = \
                np.hstack((change_frames, [i + j + 1]))

    # -------------------------------------------------
    # set up the stimulus behavior in current trial
    # -------------------------------------------------
    # extract current trial's condition
    cnd = cnd_array[itrial - 1]
    # find out in which order the images appear
    if cnd == 1 or cnd == 3:
        order = 1  # Face - House
    elif cnd == 2 or cnd == 4:
        order = 2  # House - Face
    else:
        order = None
        print('Invalid condition number!')
    print(f"Cnd: {cnd}   #Events: {n_total_evnts}   ", end="")

    # randomly decide on gap duration
    gap_dur = random.choice(gap_dur_list)

    # randomly decide on which image to cue (show in the beginning)
    cue_image = random.choice([1, 2])
    # decide on the cue/target image
    if cnd == 1 or cnd == 2:
        cue_image = 1
    elif cnd == 3 or cnd == 4:
        cue_image = 2
    else:
        cue_image = None
        print("Invalid condition number!")

    # for now: make sure the left image is tagged with f1 and the other with f2
    freq1 = 7.5
    freq2 = 12
    if order == 1:
        IRR_IMAGE1_nFRAMES = REF_RATE / freq1
        IRR_IMAGE2_nFRAMES = REF_RATE / freq2
    elif order == 2:
        IRR_IMAGE1_nFRAMES = REF_RATE / freq2
        IRR_IMAGE2_nFRAMES = REF_RATE / freq1
    else:
        IRR_IMAGE1_nFRAMES = None
        IRR_IMAGE2_nFRAMES = None
        print("Invalid image order!")

    # pick the tilting image for each event , independently of the cued image
    tilt_images = np.random.choice([1, 2], n_total_evnts)
    # pick the tilting direction for each event
    tilt_dirs = np.random.choice(['CW', 'CCW'], n_total_evnts)
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

    if acc_trial == 1:
        tilt_mag = 25
        tilt_change = 0
    else:
        # calculate what titl angle (magnitude) to use
        tilt_change = sup.cal_next_tilt(goal_perf=80, run_perf=prev_run_perf)
        tilt_mag = int(prev_tilt_mag + tilt_change)
        # take care of saturated scenarios
        if tilt_mag > 49:
            tilt_mag = 49
        elif tilt_mag < 1:
            tilt_mag = 1
    print(f"TiltAng: {(tilt_mag / 10):3.1f}deg   ", end="")

    # load the changed image
    image3_directory1cw = os.path.join("images",
                                       f"face_tilt{tilt_mag}_CW.png")
    image3_directory1ccw = os.path.join("images",
                                        f"face_tilt{tilt_mag}_CCW.png")
    image3_directory2cw = os.path.join("images",
                                       f"house_tilt{tilt_mag}_CW.png")
    image3_directory2ccw = os.path.join("images",
                                        f"house_tilt{tilt_mag}_CCW.png")

    rel_image3_1cw = visual.ImageStim(win,
                                      image=image3_directory1cw,
                                      size=IMAGE3_SIZE,
                                      opacity=IMAGE_OPACITY)
    rel_image3_1ccw = visual.ImageStim(win,
                                       image=image3_directory1ccw,
                                       size=IMAGE3_SIZE,
                                       opacity=IMAGE_OPACITY)
    rel_image3_2cw = visual.ImageStim(win,
                                      image=image3_directory2cw,
                                      size=IMAGE3_SIZE,
                                      opacity=IMAGE_OPACITY)
    rel_image3_2ccw = visual.ImageStim(win,
                                       image=image3_directory2ccw,
                                       size=IMAGE3_SIZE,
                                       opacity=IMAGE_OPACITY)
    # -------------------------------------------------
    # run the stimulus
    # -------------------------------------------------
    # set current event number
    cur_evnt_n = 0
    # ------------------
    # instruction period
    # ------------------
    # run gap period
    for igap in range(random.choice(gap_dur_list)):
        win.flip()
    # run the cue stimulus
    # randomly move the cue vertically
    cue_yoffset = random.choice(range(-5, 5))
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
        sup.draw_fixdot(win=win, size=FIX_SIZE,
                        pos=(FIX_X, FIX_Y),
                        cue=cue_image)
        win.flip()
    # ------------------
    # main period
    # ------------------
    # reset the timer in the beginning of the task (super imposed images)
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

        # if conditions satisfied change the image
        if iframe in change_frames:
            if tilt_dirs[cur_evnt_n - 1] == 'CW':
                if tilt_images[cur_evnt_n - 1] == 1:
                    rel_image3_1cw.pos = (path1_x[iframe], path1_y[iframe])
                    rel_image2.draw()
                    rel_image3_1cw.draw()
                elif tilt_images[cur_evnt_n - 1] == 2:
                    rel_image3_2cw.pos = (path2_x[iframe], path2_y[iframe])
                    rel_image3_2cw.draw()
                    rel_image1.draw()
            else:
                if tilt_images[cur_evnt_n - 1] == 1:
                    rel_image3_1ccw.pos = (path1_x[iframe], path1_y[iframe])
                    rel_image2.draw()
                    rel_image3_1ccw.draw()
                elif tilt_images[cur_evnt_n - 1] == 2:
                    rel_image3_2ccw.pos = (path2_x[iframe], path2_y[iframe])
                    rel_image3_2ccw.draw()
                    rel_image1.draw()
        # if not, show the unchanged versions
        else:
            rel_image2.draw()
            rel_image1.draw()

        # draw irrelevant images conditionally
        if sup.decide_on_show(iframe, IRR_IMAGE1_nFRAMES):
            irr_image1.draw()
        if sup.decide_on_show(iframe, IRR_IMAGE2_nFRAMES):
            irr_image2.draw()
        sup.draw_fixdot(win=win, size=FIX_SIZE,
                        pos=(FIX_X, FIX_Y),
                        cue=cue_image)
        win.flip()

        # force exit with 'escape' button
        if 'backspace' in pressed_key:
            core.quit()
        # check if spacebar is pressed within 1 sec from tilt
        if 'num_insert' in pressed_key:
            res_t = timer.getTime()
            response_times.append(round(res_t * 1000))
    response_times.pop(0)
    # evaluate the response
    [resp_eval, avg_rt] = sup.evaluate_response2(cue_image, tilt_images,
                                                 change_times, response_times)
    if np.isnan(avg_rt):
        print(f"Eval: {int(resp_eval)}   avgRT:  nan    ", end="")
    else:
        print(f"Eval: {int(resp_eval)}   avgRT: {avg_rt:03d}ms   ", end="")

    # -------------------------------------------------
    # create data frame and save
    # -------------------------------------------------
    # create a dictionary of variables to be saved
    trial_dict = {'trial_num': [acc_trial],
                  'condition_num': [cnd],
                  'cued_image': [cue_image],
                  'image_order': [order],
                  'n_events': n_total_evnts,
                  'tilted_images': [tilt_images],
                  'tilt_directions': [tilt_dirs],
                  'tilt_magnitude': [tilt_mag],
                  'avg_rt': [avg_rt],
                  'response_evaluation': [resp_eval],
                  'cummulative_performance': [np.nan],
                  'running_performance': [np.nan]}
    # convert to data frame
    dfnew = pd.DataFrame(trial_dict)
    # if not first trial, load the existing data frame and concatenate
    if acc_trial > 1:
        df = pd.read_json(data_path)
        dfnew = pd.concat([df, dfnew], ignore_index=True)
    # calculate the cumulative performance (all recorded trials)
    eval_series = dfnew.response_evaluation
    eval_array = eval_series.values
    cum_perf = round(sum(eval_array) / len(eval_array) * 100, 2)
    print(f"CumPerf: {cum_perf:6.2f}%   ", end="")
    # calculate the running performance (last 10 trials)
    run_perf = round(sum(eval_array[-10:]) / len(eval_array[-10:]) * 100, 2)
    print(f"RunPerf: {run_perf:6.2f}%")
    # fill the remaining values in the data frame
    dfnew.loc[acc_trial - 1,
    ['cummulative_performance',
     'running_performance']] = [cum_perf, run_perf]
    # save the data frame
    dfnew.to_json(data_path)

    # run the inter-trial period
    if resp_eval:
        color = 'limegreen'
    else:
        color = 'tomato'
    for igap in range(ITI_DUR):
        sup.draw_probe(win, color)
        win.flip()

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
