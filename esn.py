import sys
import matplotlib.pyplot as plt
import numpy as np
from easyesn import RegressionESN
from easyesn import helper as hlp
from env import Env

WITH_NOISE = True

def shuffle(*arrs):
    """ shuffle.

    Shuffle given arrays at unison, along first axis.

    Arguments:
        *arrs: Each array to shuffle at unison.

    Returns:
        Tuple of shuffled arrays.

    """
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    p = np.random.permutation(len(arrs[0]))
    return tuple(arr[p] for arr in arrs)

def U(t):
    if t <= 349: return 0.12
    if t <= 528: return 0.25
    if t <= 872: return 0.5
    if t <= 1042: return 0.25
    return 0.12

env = Env()
obs = env.reset()
_t = 2.
s_arr, y_arr = [], []
while True:
    obs, rew, done, info = env.step(U(_t), _t)
    y_ = info['y']
    s_ = info['s']
    s_arr.append(np.reshape(obs, (-1)))
    y_arr.append([y_])
    _t += np.random.uniform(0.1, 0.5)
    if done:
        break
    
s_arr, y_arr = shuffle(s_arr, y_arr)

if WITH_NOISE:
    s_arr *= np.random.uniform(0.95, 1.05)
    y_arr *= np.random.uniform(0.95, 1.05)
    
trainX, trainY = s_arr[:2000], y_arr[:2000]
testX, testY = s_arr[2000:2400], y_arr[2000:2400]

esn = RegressionESN(n_input=1, n_output=1, n_reservoir=50, leakingRate=1e-4, regressionParameters=[1e-2], solver="lsqr")

esn.fit(trainX, trainY, transientTime=20, verbose=1)
y_test_pred = esn.predict(testX, transientTime=2, verbose=0)
mse = (y_test_pred - testY) ** 2
print(np.mean(mse))
