# from psychopy import monitors, visual, event, core
from datetime import date, datetime
import numpy as np
import math


def test_refresh_rate(win, ref_rate):
    # measure the actual frame rate
    actual_fr = win.getActualFrameRate(nIdentical=10,
                                       nMaxFrames=100,
                                       nWarmUpFrames=10,
                                       threshold=1)
    if actual_fr is not None:
        actual_fr = round(actual_fr, 2)

    print('\n=======================================================')
    print(f"Nominal refresh rate:  {ref_rate} Hz")
    print(f"Measured refresh rate: {actual_fr} Hz\n")


def config_mon_imac24():
    monitor = monitors.Monitor('prim_mon', width=54.7, distance=57)
    monitor.setSizePix([2240, 1260])
    return monitor


def config_mon_macair():
    monitor = monitors.Monitor('prim_mon', width=33.78, distance=57)
    monitor.setSizePix([1440, 900])
    return monitor


def config_mon_dell():
    monitor = monitors.Monitor('prim_mon', width=60.45, distance=57)
    monitor.setSizePix([1920, 1080])
    return monitor


def config_win(mon, fullscr, screen):
    if fullscr:
        win = visual.Window(monitor=mon,
                            screen=screen,
                            units='deg',
                            pos=[0, 0],
                            fullscr=fullscr,
                            color=[0, 0, 0])
    else:
        win = visual.Window(monitor=mon,
                            units='deg',
                            size=[700, 700],
                            pos=[0, 0],
                            color=[0, 0, 0])
    win.mouseVisible = False
    return win


def draw_fixdot(win, size, pos, cue):
    if cue == 1:
        fix_marker = '+'
    elif cue == 2:
        fix_marker = 'x'
    else:
        fix_marker = None
        print("Invalid cue!")
    fixdot = visual.TextStim(win=win,
                             text=fix_marker,
                             height=size,
                             pos=pos,
                             color='black')
    fixdot.draw()


def draw_probe(win, color, radius=.5, pos=(0, 0)):
    inner_probe = visual.Circle(win,
                                radius=radius,
                                fillColor=color,
                                pos=pos)
    inner_probe.draw()


def get_date():
    today = date.today()
    return (str(today.year).zfill(4) +
            str(today.month).zfill(2) +
            str(today.day).zfill(2))


def get_time():
    now = datetime.now()
    return now.strftime("%H%M%S")


def block_msg(win, iblock, command_keys):
    msg = f"< Block {iblock} >" \
          f"\n\nReady to begin?"
    message = visual.TextStim(win,
                              text=msg,
                              color='black',
                              height=.5,
                              alignText='center')
    message.pos = (0, 0)
    message.draw()

    commands = '[Backspace]: Quit\t[0/Insert]: Begin'
    cmnd_text = visual.TextStim(win,
                                text=commands,
                                color='black',
                                height=.5,
                                alignText='center')
    cmnd_text.pos = (0, -2)
    cmnd_text.draw()

    win.flip()
    pressed_key = event.waitKeys(keyList=list(command_keys.values()))
    if command_keys['quit_key'] in pressed_key:
        core.quit()
    elif command_keys['response_key'] in pressed_key:
        pass

    # show a blanck window for one second
    for iframe in range(60):
        win.flip()


def decide_on_show(iframe, nframes):
    # iframe: current frame number
    # nframes: number of frames as the image interval
    iactive = math.ceil(nframes // 2)  # show the img during half of the
    # interval
    index = iframe % nframes  # calculate where we are now in the interval
    # decide whether to show or not to show the image
    if index < iactive:
        return True
    else:
        return False


def cal_next_tilt(goal_perf, run_perf):
    delta = goal_perf - run_perf
    delta_max = max([100 - goal_perf, goal_perf])
    step_max = 5
    step_change = round(delta / delta_max * step_max, 0)
    return step_change


def evaluate_response(cue_image, change_image, tilt_times, resp_times):
    """
    The stragy is give the subject a total point equal to the number of all
    tilts. Then subjects loose one point if they:
        - miss a tilt (false negative)
        - respond to no apparent tilt (false positive)
        - responsd earlier than 100 ms after tilt (anticipatory resp)
        - respond later than 1000 ms after tilt (late resp)
    :param cue_image: the cued image (one or two)
    :param change_image: an array of changed/tilted images
    :param tilt_times:  an array of the change/tilt times
    :param resp_times: an array of response times
    :return: performance and average reaction time
    """
    ind_valid_tilts = (change_image == cue_image)
    valid_tilt_times = tilt_times[ind_valid_tilts]

    n_all_tilts = len(tilt_times)
    n_valid_tilts = len(valid_tilt_times)
    n_resp = len(resp_times)
    available_pts = n_all_tilts
    iresp = 0
    itilt = 0
    lost_pts = 0
    valid_rt = []
    lost_on_iresp = np.full(n_resp, False)
    lost_on_itilt = np.full(n_valid_tilts, False)
    tilt_end_reached = False

    if n_resp == 0:
        lost_pts = n_valid_tilts
    elif n_valid_tilts == 0:
        lost_pts = n_resp
    else:
        while iresp < n_resp:
            rt = resp_times[iresp] - valid_tilt_times[itilt]
            if rt < 100:
                if not lost_on_iresp[iresp]:
                    lost_pts += 1
                lost_on_itilt[itilt] = True
                iresp += 1
            elif rt > 1000:
                if not lost_on_itilt[itilt]:
                    lost_pts += 1
                lost_on_iresp[iresp] = True
                if itilt < n_valid_tilts - 1:
                    itilt += 1
                else:
                    tilt_end_reached = True
                    iresp += 1
            else:
                if tilt_end_reached:
                    lost_pts += 1
                iresp += 1
                valid_rt.append(rt)
                if itilt < n_valid_tilts - 1:
                    itilt += 1
                else:
                    tilt_end_reached = True

        added_lost = np.sum(valid_tilt_times > resp_times[-1])
        lost_pts = lost_pts + added_lost

    if valid_rt:
        avg_rt = np.mean(np.array(valid_rt))
    else:
        avg_rt = np.nan

    earned_points = n_all_tilts - lost_pts
    if earned_points < 0:
        earned_points = 0
    perf = int(np.round(earned_points / available_pts * 100))

    return [perf, avg_rt]
