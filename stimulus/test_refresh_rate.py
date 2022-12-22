
import supplements as sup

ref_rate = 60

# configure the monitor and the stimulus window
mon = sup.config_mon_dell()
win = sup.config_win(mon=mon, fullscr=False, screen=9)
sup.test_refresh_rate(win, ref_rate)