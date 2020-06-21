import multiprocessing as mp
import numpy as np
import logging
import os
import sys
from env import Env
import ppo2 as network
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

S_DIM = [5, 8]
A_DIM = 10
ACTOR_LR_RATE =1e-4
CRITIC_LR_RATE = 1e-3
NUM_AGENTS = 4
TRAIN_SEQ_LEN = 1500  # take as a train batch
TRAIN_EPOCH = 1000000
MODEL_SAVE_INTERVAL = 30
RANDOM_SEED = 42
RAND_RANGE = 10000
SUMMARY_DIR = './results'
MODEL_DIR = './models'
TRAIN_TRACES = './cooked_traces/'
TEST_LOG_FOLDER = './test_results/'
LOG_FILE = './results/log'

# create result directory
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

NN_MODEL = sys.argv[1]

def main():
    env = Env()
    with tf.Session() as sess:
        actor = network.Network(sess,
                                state_dim=S_DIM, action_dim=A_DIM,
                                learning_rate=ACTOR_LR_RATE)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1000)  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            # print("Model restored.")

        time_stamp = 0

        obs = env.reset()
        s_batch, a_batch, p_batch, r_batch = [], [], [], []
        for step in range(TRAIN_SEQ_LEN):
            s_batch.append(obs)

            action_prob = actor.predict(
                np.reshape(obs, (-1, S_DIM[0], S_DIM[1])))
            
            noise = np.random.gumbel(size=len(action_prob))
            act = np.argmax(np.log(action_prob) + noise)

            obs, rew, done, info = env.step(act)
            print(0.1 + act / A_DIM * 0.4)
            action_vec = np.zeros(A_DIM)
            action_vec[act] = 1
            a_batch.append(action_vec)
            r_batch.append(rew)
            p_batch.append(action_prob)
            if done:
                break

if __name__ == '__main__':
    main()
