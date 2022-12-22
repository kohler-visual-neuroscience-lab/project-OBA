import numpy as np
from supplements import evaluate_response2

cue_image = 1
change_image = np.array([1, 2, 2, 2])
change_times = np.array([200, 400, 500, 700])
response_times = np.array([320.34])

[resp_eval, rt] = evaluate_response2(cue_image, change_image,
                                     change_times, response_times)
print((resp_eval, rt))
