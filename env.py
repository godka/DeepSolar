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
    
    def I(self, t, noise=False):
        val = -1. / (700. * 700.) * (t-700) ** 2. + 1.
        if noise:
            val *= np.random.uniform(0.95, 1.05)
        return val

    def Y(self, u_, noise=False):
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
        self.t = 1.
        self.Y(0.14)
        state = np.zeros([S_INFO, S_LEN])
        state[0, -1] = self.y[-1]
        state[1, -1] = self.y[-2]
        state[2, -1] = self.u[-1]
        state[3, -1] = self.u[-2]
        state[4, -1] = self.t / 1400.
        self.state = state
        self.t += 1
        return state

    def step(self, act, tick = None, noise = False):
        u = 0.14 + act / A_DIM * 0.4
        if noise:
            u *= np.random.uniform(0.95, 1.05)
        rew = self.Y(u, noise)
        state = np.roll(self.state, -1, axis=1)
        state[0, -1] = self.y[-1]
        state[1, -1] = self.y[-2]
        state[2, -1] = self.u[-1]
        state[3, -1] = self.u[-2]
        state[4, -1] = self.t / 1400.
        self.state = state
        if tick is None:
            self.t += 1
        else:
            self.t = tick
        rew = self.Y(u)
        smo = np.abs(self.u[-1] - self.u[-2])
        done = False
        if self.t > 1400:
            done = True
        info = {}
        info['y'] = rew
        info['s'] = [self.y[-1], self.y[-2], self.u[-1], self.u[-2], self.t / 1400.]
        return self.state, rew - u - smo, done, info


    def stepv2(self, act, tick = None, noise = False):
        u = act
        if noise:
            u *= np.random.uniform(0.95, 1.05)
        rew = self.Y(u, noise)
        state = np.roll(self.state, -1, axis=1)
        state[0, -1] = self.y[-1]
        state[1, -1] = self.y[-2]
        state[2, -1] = self.u[-1]
        state[3, -1] = self.u[-2]
        state[4, -1] = self.t / 1400.
        self.state = state
        if tick is None:
            self.t += 1
        else:
            self.t = tick
        rew = self.Y(u)
        smo = np.abs(self.u[-1] - self.u[-2])
        done = False
        if self.t > 1400:
            done = True
        info = {}
        info['y'] = rew
        info['s'] = [self.y[-1], self.y[-2], self.u[-1], self.u[-2], self.t / 1400.]
        return self.state, rew - u - smo, done, info

        
