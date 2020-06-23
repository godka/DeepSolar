import os
import sys
import numpy as np
import h5py
import tensorflow as tf
import tflearn
from env import Env
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import plot, savefig

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
S_INFO = 5
S_LEN = 8
WITH_NOISE = True

# you can use mish active function instead
# the total performance will be improved a little bit.
def mish(x):
    return x * tf.nn.tanh(tf.nn.softplus(x))

def create_network():
    with tf.variable_scope('mlp'):
        inputs = tflearn.input_data(shape=[None, S_INFO, S_LEN])
        net = tflearn.conv_1d(inputs, 128, 5, activation='relu')
        net = tflearn.fully_connected(net, 64, activation='relu')
        net = tflearn.fully_connected(net, 32, activation='relu')
        out = tflearn.fully_connected(net, 1, activation='sigmoid')
        # net = tflearn.fully_connected(inputs, 128, activation=mish)
        # net = tflearn.fully_connected(inputs, 64, activation=mish)
        # net = tflearn.fully_connected(inputs, 128, activation=mish)
        # out = tflearn.fully_connected(net, 1, activation='sigmoid')

        return inputs, out

def U(t):
    if t <= 349: return 0.12
    if t <= 528: return 0.25
    if t <= 872: return 0.5
    if t <= 1042: return 0.25
    return 0.12

def main():
    env = Env()
    obs = env.reset()
    _t = 2.
    s_arr, y_arr = [], []
    while True:
        obs, rew, done, info = env.step(U(_t), noise=True)
        y_ = info['y']
        s_arr.append(obs)
        y_arr.append([y_])
        if done:
            break

    s_arr, y_arr = np.array(s_arr), np.array(y_arr)

    testX, testY = s_arr, y_arr
    
    _input, _output = create_network()
    network = tflearn.regression(_output, optimizer='adam', learning_rate=1e-4,
                                    loss='mean_square', name='target')

    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir = './results/')
    model.load('model/1dcnn.tfl')
    predY = model.predict(testX)
    plt.plot(testY, '--', color='red', label='Groud Truth')
    plt.plot(predY, '-', color='black', label='1D-CNN')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()