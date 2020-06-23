import os
import sys
import numpy as np
import h5py
import tensorflow as tf
import tflearn
from env import Env

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
        net = tflearn.lstm(inputs, 32, activation=mish)
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
        obs, rew, done, info = env.step(U(_t), _t, WITH_NOISE)
        y_ = info['y']
        s_arr.append(obs)
        y_arr.append([y_])
        _t += np.random.uniform(0.1, 0.5)
        if done:
            break
        
    s_arr, y_arr = tflearn.data_utils.shuffle(s_arr, y_arr)

    trainX, trainY = s_arr[:2000], y_arr[:2000]
    testX, testY = s_arr[2000:2400], y_arr[2000:2400]
    
    _input, _output = create_network()
    network = tflearn.regression(_output, optimizer='adam', learning_rate=1e-4,
                                    loss='mean_square', name='target')

    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir = './results/')
    model.fit(trainX, trainY, n_epoch=50, 
            batch_size=32,
            validation_set=(testX, testY),
            show_metric=False, shuffle=True, run_id='lstm')
    model.save("model/lstm.tfl")

if __name__ == '__main__':
    main()