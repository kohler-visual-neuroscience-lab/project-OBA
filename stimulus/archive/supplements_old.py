from psychopy import monitors, visual, event, core
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


def escape_session():
    exit_key = event.getKeys(keyList=['backspace'])
    if 'backspace' in exit_key:
        core.quit()


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
    if 'backspace' in pressed_key:
        core.quit()
    elif 'num_insert' in pressed_key:
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


def evaluate_response(cue_image, change_image, rt, response):
    if (cue_image == change_image) and (rt >= 0):
        resp = True
    elif (cue_image != change_image) and (response == 0):
        resp = True
    else:
        resp = False
    return resp


def cal_next_tilt(goal_perf, run_perf):
    delta = goal_perf - run_perf
    delta_max = max([100 - goal_perf, goal_perf])
    step_max = 5
    step_change = round(delta / delta_max * step_max, 0)
    return step_change
