from psychopy import monitors, visual, event, core
from datetime import date, datetime
import math


def test_refresh_rate(win, ref_rate):
    # measure the actual frame rate
    actual_fr = win.getActualFrameRate(nIdentical=10,
                                       nMaxFrames=100,
                                       nWarmUpFrames=10,
                                       threshold=1)
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
                            size=[500, 500],
                            pos=[0, 0],
                            color=[0, 0, 0])
    win.mouseVisible = False
    return win


def draw_fixdot(win, size, pos):
    fixdot = visual.TextStim(win=win,
                             text='+',
                             height=size,
                             pos=pos,
                             color='black')
    fixdot.draw()


def escape_session():
    exit_key = event.getKeys(keyList=['escape'])
    if 'escape' in exit_key:
        core.quit()


def get_date():
    today = date.today()
    return (str(today.year).zfill(4) +
            str(today.month).zfill(2) +
            str(today.day).zfill(2))


def get_time():
    now = datetime.now()
    return now.strftime("%H%M%S")


def block_msg(win, iblock):
    msg = f"< Block {iblock} >" \
          f"\n\nReady to begin?"
    message = visual.TextStim(win,
                              text=msg,
                              color='black',
                              height=.5,
                              alignText='center')
    message.pos = (0, 0)
    message.draw()

    commands = '[Escape]: Cancel\t[Space]: Begin'
    cmnd_text = visual.TextStim(win,
                                text=commands,
                                color='black',
                                height=.5,
                                alignText='center')
    cmnd_text.pos = (0, -2)
    cmnd_text.draw()

    win.flip()
    pressed_key = event.waitKeys(keyList=['space', 'escape'])
    if 'escape' in pressed_key:
        core.quit()
    elif 'space' in pressed_key:
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
