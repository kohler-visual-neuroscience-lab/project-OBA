
import numpy as np


def eval_resp(cue_image, change_image, tilt_times, resp_times):
    """
    Mohammad Shams <m.shams.ahmar@gmail.com>
    Dec 27, 2022

    The stragy is give the subject a total point equal to the number of all
    tilts. Then subjects loose one point if they:
        - miss a tilt (false negative)
        - respond to no apparent tilt (false positive)
        - responsd earlier than 200 ms after tilt (anticipatory resp)
        - respond later than 1000 ms after tilt (late resp)
    :param cue_image: the cued image (one or two)
    :param change_image: an array of changed/tilted image
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
            if rt < 200:
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
        avg_rt = np.round(np.mean(np.array(valid_rt)))
    else:
        avg_rt = np.nan

    earned_points = n_all_tilts - lost_pts
    if earned_points < 0:
        earned_points = 0
    perf = int(np.round(earned_points / available_pts * 100))

    return [perf, avg_rt]
