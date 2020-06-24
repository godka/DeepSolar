import numpy as np
import logging
import os
import sys
from env import Env

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

S_DIM = [5, 8]
A_DIM = 10
RANDOM_SEED = 42
RAND_RANGE = 1000
TRAIN_SEQ_LEN = 1500

def U(t):
    if t <= 349: return 0.12
    if t <= 528: return 0.25
    if t <= 872: return 0.5
    if t <= 1042: return 0.25
    return 0.12

def main():
    env = Env()
    
    arr = []
    f = open('1.csv', 'r')
    for line in f:
        arr.append(float(line))
    f.close()
    
    obs = env.reset()
    s_batch, a_batch, p_batch, r_batch = [], [], [], []
    for step in range(TRAIN_SEQ_LEN):
        s_batch.append(obs)
        act = int((arr[step] - 0.14) / 0.4 * A_DIM)
        obs, rew, done, info = env.step(act)
        r_batch.append(info['y'])
        if done:
            break
    print(np.mean(r_batch), np.mean(arr), np.mean(np.abs(np.diff(arr))))

    arr0 = []
    for p in range(1400):
        arr0.append(U(p))

    obs = env.reset()
    s_batch, a_batch, p_batch, r_batch = [], [], [], []
    for step in range(TRAIN_SEQ_LEN):
        s_batch.append(obs)
        obs, rew, done, info = env.stepv2(arr0[step])
        r_batch.append(info['y'])
        if done:
            break
    print(np.mean(r_batch), np.mean(arr0), np.mean(np.abs(np.diff(arr0))))

    arr = []
    f = open('3.csv', 'r')
    for line in f:
        arr.append(float(line))
    f.close()
    
    obs = env.reset()
    s_batch, a_batch, p_batch, r_batch = [], [], [], []
    for step in range(TRAIN_SEQ_LEN):
        s_batch.append(obs)
        act = int((arr[step] - 0.14) / 0.4 * A_DIM)
        obs, rew, done, info = env.step(act)
        r_batch.append(info['y'])
        if done:
            break
    print(np.mean(r_batch), np.mean(arr), np.mean(np.abs(np.diff(arr))))

if __name__ == '__main__':
    main()
