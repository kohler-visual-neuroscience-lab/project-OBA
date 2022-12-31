from psychopy import monitors, visual
import numpy as np

ref_rate = 120

# configure the monitor and the stimulus window
monitor = monitors.Monitor('prim_mon', width=54.7, distance=57)
monitor.setSizePix([2240, 1260])

win = visual.Window(monitor=monitor,
                    units='deg',
                    size=[500, 500],
                    pos=[0, 0],
                    color=[0, 0, 0])

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

# -------------------------------
# flash in one-second intervals
first_row = np.arange(0, stop=5*ref_rate, step=ref_rate)
second_row = first_row + 1
mat = np.vstack([first_row, second_row])
for i in range(8):
    new_row = mat[-1, :] + 1
    mat = np.vstack([mat, new_row])
mat.flatten()
mat.sort()

for i in range(5 * ref_rate):
    if i in mat:
        probe = visual.Circle(win,
                              radius=1,
                              fillColor='tomato',
                              pos=(0, 0))
        probe.draw()
    win.flip()
