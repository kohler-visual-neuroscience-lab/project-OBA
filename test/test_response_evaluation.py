import numpy as np
from lib.evaluate_responses import eval_resp

n_tests = 14
perf_arr = []
rt_arr = []
expected_perf = np.full(n_tests, np.nan, dtype=int)

# TEST #1
cue_image = 1
change_image = np.array([2, 1, 2])
change_times = np.array([2000, 5000, 7000])
response_times = np.array([])
perf, rt = eval_resp(cue_image, change_image,
                     change_times, response_times)
perf_arr.append(perf)
rt_arr.append(rt)
expected_perf[0] = 67

# TEST #2
cue_image = 1
change_image = np.array([2])
change_times = np.array([2000])
response_times = np.array([])
perf, rt = eval_resp(cue_image, change_image,
                     change_times, response_times)
perf_arr.append(perf)
rt_arr.append(rt)
expected_perf[1] = 100

# TEST #3
cue_image = 2
change_image = np.array([2])
change_times = np.array([2000])
response_times = np.array([])
perf, rt = eval_resp(cue_image, change_image,
                     change_times, response_times)
perf_arr.append(perf)
rt_arr.append(rt)
expected_perf[2] = 0

# TEST #4
cue_image = 1
change_image = np.array([1])
change_times = np.array([2000])
response_times = np.array([2500, 4000])
perf, rt = eval_resp(cue_image, change_image,
                     change_times, response_times)
perf_arr.append(perf)
rt_arr.append(rt)
expected_perf[3] = 0

# TEST #5
cue_image = 1
change_image = np.array([1, 1])
change_times = np.array([1000, 3000])
response_times = np.array([2500])
perf, rt = eval_resp(cue_image, change_image,
                     change_times, response_times)
perf_arr.append(perf)
rt_arr.append(rt)
expected_perf[4] = 0

# TEST #6
cue_image = 1
change_image = np.array([1, 1])
change_times = np.array([1000, 3000])
response_times = np.array([2500, 3200])
perf, rt = eval_resp(cue_image, change_image,
                     change_times, response_times)
perf_arr.append(perf)
rt_arr.append(rt)
expected_perf[5] = 50

# TEST #7
cue_image = 1
change_image = np.array([1, 1, 1])
change_times = np.array([1000, 3000, 5000])
response_times = np.array([1500, 2500, 5200, 5300])
perf, rt = eval_resp(cue_image, change_image,
                     change_times, response_times)
perf_arr.append(perf)
rt_arr.append(rt)
expected_perf[6] = 33

# TEST #8
cue_image = 1
change_image = np.array([1, 1, 1])
change_times = np.array([1000, 3000, 5000])
response_times = np.array([1500, 4200, 5200])
perf, rt = eval_resp(cue_image, change_image,
                     change_times, response_times)
perf_arr.append(perf)
rt_arr.append(rt)
expected_perf[7] = 67

# TEST #9
cue_image = 1
change_image = np.array([2, 2, 1])
change_times = np.array([3000, 5000, 7000])
response_times = np.array([2500, 5200, 5300])
perf, rt = eval_resp(cue_image, change_image,
                     change_times, response_times)
perf_arr.append(perf)
rt_arr.append(rt)
expected_perf[8] = 0

# TEST #10
cue_image = 2
change_image = np.array([1, 2, 2, 2])
change_times = np.array([1000, 3000, 5000, 7000])
response_times = np.array([3400, 6100, 7600])
perf, rt = eval_resp(cue_image, change_image,
                     change_times, response_times)
perf_arr.append(perf)
rt_arr.append(rt)
expected_perf[9] = 75

# TEST #11
cue_image = 2
change_image = np.array([2, 2, 1])
change_times = np.array([1000, 2500, 7000])
response_times = np.array([1500, 7400, 7500])
perf, rt = eval_resp(cue_image, change_image,
                     change_times, response_times)
perf_arr.append(perf)
rt_arr.append(rt)
expected_perf[10] = 33

# TEST #12
cue_image = 1
change_image = np.array([1, 2, 1, 2])
change_times = np.array([1000, 3000, 5000, 7000])
response_times = np.array([])
perf, rt = eval_resp(cue_image, change_image,
                     change_times, response_times)
perf_arr.append(perf)
rt_arr.append(rt)
expected_perf[11] = 50

# TEST #13
cue_image = 1
change_image = np.array([1, 2, 1, 2])
change_times = np.array([1000, 3000, 5000, 7000])
response_times = np.array([2100, 3000, 5500])
perf, rt = eval_resp(cue_image, change_image,
                     change_times, response_times)
perf_arr.append(perf)
rt_arr.append(rt)
expected_perf[12] = 50

# TEST #14
cue_image = 1
change_image = np.array([1, 1, 1, 1])
change_times = np.array([1000, 3000, 5000, 7000])
response_times = np.array([2100, 3000, 5500])
perf, rt = eval_resp(cue_image, change_image,
                     change_times, response_times)
perf_arr.append(perf)
rt_arr.append(rt)
expected_perf[13] = 25

perf_arr = np.array(perf_arr)
rt_arr = np.round(np.array(rt_arr))
comp_arr = expected_perf == perf_arr
test = np.all(comp_arr)
failed_tests = np.nonzero(comp_arr == 0)[0] + 1
if test:
    print("*** TEST PASSED! ***")
    print(f"average RTs in each trial/test = {rt_arr}")
else:
    print("### MISMATCH FOUND ON TEST(S):")
    print(failed_tests)
    print(f"expected perf =  {expected_perf}")
    print(f"evaluated perf = {perf_arr}")
