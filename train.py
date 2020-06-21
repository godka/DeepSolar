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
NUM_AGENTS = 16
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
PPO_TRAINING_EPO = 5
# create result directory
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

NN_MODEL = None

def central_agent(net_params_queues, exp_queues):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS
    tf_config=tf.ConfigProto(intra_op_parallelism_threads=5,
                            inter_op_parallelism_threads=5)
    with tf.Session(config = tf_config) as sess, open(LOG_FILE + '_test.txt', 'w') as test_log_file:
        summary_ops, summary_vars = build_summaries()

        actor = network.Network(sess,
                state_dim=S_DIM, action_dim=A_DIM,
                learning_rate=ACTOR_LR_RATE)

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver(max_to_keep=1000)  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")
        
        max_reward, max_epoch = -10000., 0
        tick_gap = 0
        # while True:  # assemble experiences from agents, compute the gradients
        for epoch in range(TRAIN_EPOCH):
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put(actor_net_params)

            s, a, p, g, r = [], [], [], [], []
            for i in range(NUM_AGENTS):
                s_, a_, p_, g_, r_ = exp_queues[i].get()
                s += s_
                a += a_
                p += p_
                g += g_
                r += r_
            s_batch = np.stack(s, axis=0)
            a_batch = np.vstack(a)
            p_batch = np.vstack(p)
            v_batch = np.vstack(g)
            r_batch = np.vstack(r)

            for _ in range(PPO_TRAINING_EPO):
                actor.train(s_batch, a_batch, p_batch, v_batch, epoch)
            # actor.train(s_batch, a_batch, v_batch, epoch)
            
            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                                       str(epoch) + ".ckpt")

                avg_reward = np.mean(r_batch)

                if avg_reward > max_reward:
                    max_reward = avg_reward
                    max_epoch = epoch
                    tick_gap = 0
                else:
                    tick_gap += 1
                
                if tick_gap >= 10:
                    # saver.restore(sess, SUMMARY_DIR + "/nn_model_ep_" + str(max_epoch) + ".ckpt")
                    actor.set_entropy_decay()
                    tick_gap = 0

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: actor.get_entropy(epoch),
                    summary_vars[1]: avg_reward,
                    summary_vars[2]: 0.
                })

                writer.add_summary(summary_str, epoch)
                writer.flush()

def agent(agent_id, net_params_queue, exp_queue):
    env = Env(agent_id)
    with tf.Session() as sess, open(SUMMARY_DIR + '/log_agent_' + str(agent_id), 'w') as log_file:
        actor = network.Network(sess,
                                state_dim=S_DIM, action_dim=A_DIM,
                                learning_rate=ACTOR_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)

        time_stamp = 0

        for epoch in range(TRAIN_EPOCH):
            obs = env.reset()
            s_batch, a_batch, p_batch, r_batch = [], [], [], []
            for step in range(TRAIN_SEQ_LEN):
                s_batch.append(obs)

                action_prob = actor.predict(
                    np.reshape(obs, (-1, S_DIM[0], S_DIM[1])))
                
                noise = np.random.gumbel(size=len(action_prob))
                act = np.argmax(np.log(action_prob) + noise)

                obs, rew, done, info = env.step(act)

                action_vec = np.zeros(A_DIM)
                action_vec[act] = 1
                a_batch.append(action_vec)
                r_batch.append(rew)
                p_batch.append(action_prob)
                if done:
                    break
            v_batch = actor.compute_v(s_batch, a_batch, r_batch, done)
            exp_queue.put([s_batch, a_batch, p_batch, v_batch, r_batch])

            actor_net_params = net_params_queue.get()
            actor.set_network_params(actor_net_params)

def build_summaries():
    td_loss = tf.Variable(0.)
    tf.summary.scalar("Beta", td_loss)
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", eps_total_reward)
    entropy = tf.Variable(0.)
    tf.summary.scalar("Entropy", entropy)

    summary_vars = [td_loss, eps_total_reward, entropy]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def main():

    np.random.seed(RANDOM_SEED)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in range(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
