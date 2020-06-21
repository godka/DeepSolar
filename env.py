import numpy as np
import os

RANDOM_SEED = 42
S_INFO = 5
S_LEN = 8
A_DIM = 10

class Env(object):
    def __init__(self, seed = RANDOM_SEED):
        np.random.seed(seed)
        self.t = 0
        self.u = [0.1]
        self.y = [0., 0.]
        self.state = None
    
    def I(self, t):
        val = -1. / (720. * 720.) * (t-720) ** 2. + 1.
        return val

    def Y(self, u_):
        self.u.append(u_)
        d = [0.3, 0.6, 2.0, 1.3]
        yt_1 = self.y[-1]
        yt_2 = self.y[-2]
        Ut_1 = self.u[-1]
        Ut_2 = self.u[-2]
        It_2 = self.I(self.t - 2.)
        y_ = (1-d[1]) * yt_1 + \
            (1-d[3]) * yt_1 * Ut_1 / Ut_2 + \
            (d[3]-1) * (1.+d[1]) * yt_2 * Ut_1 / Ut_2 + \
            d[0] * d[2] * Ut_1 * It_2 - \
            d[0] * Ut_1 * yt_1 + \
            d[0] * (1.+d[1]) * Ut_1 * yt_2
        self.y.append(y_)
        return y_

    def reset(self):
        self.t = 2.
        self.Y(0.1)
        state = np.zeros([S_INFO, S_LEN])
        state[0, -1] = self.y[-1]
        state[1, -1] = self.y[-2]
        state[2, -1] = self.u[-1]
        state[3, -1] = self.u[-2]
        state[4, -1] = self.t / 1440.
        self.state = state
        self.t += 1
        return state

    def step(self, act):
        u = 0.1 + act / A_DIM * 0.4
        rew = self.Y(u)
        state = np.roll(self.state, -1, axis=1)
        state[0, -1] = self.y[-1]
        state[1, -1] = self.y[-2]
        state[2, -1] = self.u[-1]
        state[3, -1] = self.u[-2]
        state[4, -1] = self.t / 1440.
        self.state = state
        self.t += 1
        rew = self.Y(u)
        smo = np.abs(self.u[-1] - self.u[-2])
        done = False
        if self.t > 1440:
            done = True
        return self.state, rew - u - smo, done, {}


        