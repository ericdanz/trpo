from utils import *
import numpy as np
import random
import tensorflow as tf
import time
import os
import logging
import gym
from gym import envs, scoreboard
from gym.spaces import Discrete, Box
import prettytensor as pt
from space_conversion import SpaceConversionEnv
import tempfile
import sys
import datetime
import subprocess


def policy_model(input_layer,hidden_layer_sizes=[64,64],task_size=False,output_size=3):
    net = input_layer
    for i,hidden_size in enumerate(hidden_layer_sizes):
        with tf.variable_scope('h{}'.format(i)):
            W = tf.Variable(tf.zeros([net.get_shape()[1].value,hidden_size]))
            b = tf.Variable(tf.zeros([hidden_size]))
            h = tf.nn.bias_add(tf.matmul(net,W),b)
            a = tf.nn.tanh(h)
        net = a

    with tf.variable_scope('out_mean'):
        W = tf.Variable(tf.zeros([net.get_shape()[1].value,output_size]))
        b = tf.Variable(tf.zeros([output_size]))
        h3 = tf.nn.bias_add(tf.matmul(net,W),b)
    #fixed std dev
    with tf.variable_scope('out_std'):
        W = tf.Variable(tf.zeros([1,output_size]))
        h3_std = tf.tile(tf.exp(W),tf.pack([tf.shape(h3)[0],1]))
    h3 = tf.reshape(h3,[-1,1,output_size])
    h3_std = tf.reshape(h3_std,[-1,1,output_size])
    output = tf.concat(1,[h3,h3_std])

    return output

def loglikelihood(actions,action_dist,dims):
    mean_n = tf.reshape(action_dist[:,0,:],[tf.shape(action_dist)[0],tf.shape(action_dist)[2]])
    std_n = tf.reshape(action_dist[:,1,:],[tf.shape(action_dist)[0],tf.shape(action_dist)[2]])
    return -0.5 * tf.reduce_sum(tf.square((actions-mean_n) / std_n),1) \
                -0.5 * tf.log(2.0*np.pi)*dims - tf.reduce_sum(tf.log(std_n),1)

def kl_divergence(action_dist,oldaction_dist,dims):
    mean_n = tf.reshape(action_dist[:,0,:],[tf.shape(action_dist)[0],tf.shape(action_dist)[2]])
    std_n = tf.reshape(action_dist[:,1,:],[tf.shape(action_dist)[0],tf.shape(action_dist)[2]])
    oldmean_n = tf.reshape(oldaction_dist[:,0,:],[tf.shape(oldaction_dist)[0],tf.shape(oldaction_dist)[2]])
    oldstd_n = tf.reshape(oldaction_dist[:,1,:],[tf.shape(oldaction_dist)[0],tf.shape(oldaction_dist)[2]])
    return (tf.reduce_sum(tf.log(std_n/oldstd_n),1) + tf.reduce_sum((tf.square(oldstd_n)+
        tf.square(oldmean_n - mean_n)) / (2.0 * tf.square(std_n)),1) - 0.5*dims)

def compute_advantage(vf, paths, gamma=.999, lam=.99):
    # Compute return, baseline, advantage
    for path in paths:
        path["returns"] = discount(path["rewards"], gamma)
        b = path["baseline"] = vf.predict(path)
        b1 = np.append(b, 0 if path["terminated"] else b[-1])
        deltas = path["rewards"] + gamma*b1[1:] - b1[:-1]
        path["advantage"] = discount(deltas, gamma * lam)
    alladv = np.concatenate([path["advantage"] for path in paths])
    # Standardize advantage
    std = alladv.std()
    mean = alladv.mean()
    for path in paths:
        path["advantage"] = (path["advantage"] - mean) / std


class TRPOAgent(object):

    config = dict2(**{
        "timesteps_per_batch": 15000,
        "max_pathlength": 500,
        "max_kl": 0.01,
        "cg_damping": 0.1,
        "gamma": 0.95})

    def __init__(self, env):
        self.env = env
        if not isinstance(env.observation_space, Box):
            print("Incompatible obs space.")
            exit(-1)
        print("Observation Space", env.observation_space)
        print("Action Space", env.action_space)
        self.session = tf.Session()
        self.end_count = 0
        self.train = True
        self.prev_obs = np.zeros((1, env.observation_space.shape[0]))
        self.advant = advant = tf.placeholder(dtype, shape=[None], name="advant")
        self.beta = .01
        self.learning_rate = tf.placeholder(dtype,shape=(),name="learning_rate")
        self.clip = 5
        self.eps = 1e-8
        self.count = None
        self.mean = None
        self.std = None
        with subprocess.Popen(["git","rev-parse" ,"HEAD"], stdout=subprocess.PIPE) as proc:
            GIT_COMMIT = str(proc.stdout.read())[-8:-3]
        now = datetime.datetime.now()
        identifying_string = "{}-{}-{}-rand{}:".format(now.day,now.month,now.year,str(time.time())[-3:]) + GIT_COMMIT
        self.train_writer = tf.train.SummaryWriter('./train/'+identifying_string,
                                      self.session.graph)
        if isinstance(env.action_space,Discrete):
            self.discrete = True
        else:
            self.discrete = False
        if self.discrete:
            self.obs = obs = tf.placeholder(
                dtype, shape=[
                    None, 2 * env.observation_space.shape[0] + env.action_space.n], name="obs")
            self.dimensions = env.action_space.n
            self.prev_action = np.zeros((1, env.action_space.n))
            self.oldaction_dist = oldaction_dist = tf.placeholder(dtype, shape=[None, env.action_space.n], name="oldaction_dist")
            self.action = action = tf.placeholder(tf.int64, shape=[None], name="action")
            action_dist_n, _ = (pt.wrap(self.obs).
                                fully_connected(64, activation_fn=tf.nn.tanh).
                                softmax_classifier(env.action_space.n))

            self.action_dist_n = action_dist_n
            N = tf.shape(obs)[0]
            p_n = slice_2d(action_dist_n, tf.range(0, N), action)
            oldp_n = slice_2d(oldaction_dist, tf.range(0, N), action)
            ratio_n = p_n / oldp_n
            Nf = tf.cast(N, dtype)
            surr = -tf.reduce_mean(ratio_n * advant)  # Surrogate loss
            kl = tf.reduce_sum(oldaction_dist * tf.log((oldaction_dist + self.eps) / (action_dist_n + self.eps))) / Nf
            ent = tf.reduce_sum(-action_dist_n * tf.log(action_dist_n + self.eps)) / Nf
        else:
            self.obs = obs = tf.placeholder(
                dtype, shape=[
                    None, 2 * env.observation_space.shape[0] + env.action_space.shape[0]], name="obs")
            self.dimensions = env.action_space.shape[0]
            self.prev_action = np.zeros((1, env.action_space.shape[0]))
            self.oldaction_dist = oldaction_dist = tf.placeholder(dtype, shape=[None, 2,env.action_space.shape[0]], name="oldaction_dist")
            action_dist_n = policy_model(self.obs,
                                        hidden_layer_sizes=[60,60],
                                        output_size=self.dimensions)
            self.action = action = tf.placeholder(dtype, shape=[None,env.action_space.shape[0]], name="action")
            self.action_dist_n = action_dist_n
            N = tf.shape(obs)[0]
            Nf = tf.cast(N, dtype)
            logp_n = loglikelihood(action,action_dist_n,self.dimensions)
            oldlogp_n = loglikelihood(action,oldaction_dist,self.dimensions)
            surr = -tf.reduce_sum( tf.exp(logp_n - oldlogp_n) * advant) / Nf
            kl = tf.reduce_mean(kl_divergence(action_dist_n,oldaction_dist,self.dimensions))
            ent = tf.reduce_mean(tf.reduce_sum(tf.log(action_dist_n[:,1,:]),1) + .5  *
            np.log(2*np.pi*np.e) * self.dimensions)

        # Create neural network.


        self.losses = [surr, kl, ent]
        self.proximal_loss = surr + self.beta * kl
        self.learning_rate_value = 1e-1
        # self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.proximal_loss)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate,beta1=.1,beta2=.1).minimize(self.proximal_loss)
        #see if separating them does anything to learning
        self.surr_train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(surr)
        self.kl_train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.beta * kl)
        self.vf = VF(self.session)
        self.kl_running_avg = 0
        self.surr_running_avg = 0
        self.session.run(tf.initialize_all_variables())

    def act(self, obs, *args):
        #filter the observations
        obs = self.obs_filter(obs)
        obs = np.expand_dims(obs, 0)
        if self.prev_obs[0,0] == 0:
            self.prev_obs = obs
        obs_new = np.concatenate([obs, self.prev_obs, self.prev_action], 1)

        action_dist_n = self.session.run(self.action_dist_n, {self.obs: obs_new})
        if self.discrete:
            if self.train:
                action = int(cat_sample(action_dist_n)[0])
            else:
                action = int(np.argmax(action_dist_n))
            self.prev_action *= 0.0
            self.prev_action[0, action] = 1.0
        else:
            action = np.random.randn(self.dimensions) * action_dist_n[0,1,:] + action_dist_n[0,0,:]
            self.prev_action = np.expand_dims(np.copy(action),0)
        self.prev_obs = obs
        return action, action_dist_n, np.squeeze(obs_new)

    def obs_filter(self,obs):
        if not self.count:
            self.count = 1
        if self.count == 1:
            self.mean = obs
            self.std = np.square(self.mean)
            self.std_val = self.std
        else:
            old_mean = np.copy(self.mean)
            self.mean = old_mean + (obs - old_mean)/self.count
            self.std_val = self.std_val + (obs - old_mean)*(obs - self.mean)
            self.std = np.sqrt(self.std_val/(self.count-1))

        self.count += 1
        obs = obs - self.mean
        obs = obs / (self.std+self.eps)
        obs = np.clip(obs,-self.clip,self.clip)
        return obs

    def learn(self):
        config = self.config
        start_time = time.time()
        numeptotal = 0
        i = 0
        while True:
            # Generating paths.
            print("Rollout")
            paths = rollout(
                self.env,
                self,
                config.max_pathlength,
                config.timesteps_per_batch)

            # Computing returns and estimating advantage function.
            # for path in paths:
            #     path["baseline"] = self.vf.predict(path)
            #     path["returns"] = discount(path["rewards"], config.gamma)
            #     path["advant"] = path["returns"] - path["baseline"]
            compute_advantage(self.vf,paths,gamma=.995,lam=.97)
            # Updating policy.
            action_dist_n = np.concatenate([path["action_dists"] for path in paths])
            obs_n = np.concatenate([path["obs"] for path in paths])
            action_n = np.concatenate([path["actions"] for path in paths])
            baseline_n = np.concatenate([path["baseline"] for path in paths])
            returns_n = np.concatenate([path["returns"] for path in paths])

            # Standardize the advantage function to have mean=0 and std=1.
            advant_n = np.concatenate([path["advantage"] for path in paths])
            # advant_n -= advant_n.mean()
            #
            # # Computing baseline function for next iter.
            #
            # advant_n /= (advant_n.std() + self.eps)

            feed = {self.obs: obs_n,
                    self.action: action_n,
                    self.advant: advant_n,
                    self.oldaction_dist: action_dist_n,
                    self.learning_rate : self.learning_rate_value,
                    }


            episoderewards = np.array(
                [path["rewards"].sum() for path in paths])

            print("\n********** Iteration %i ************" % i)
            if self.env.spec.reward_threshold and \
                episoderewards.mean() > 1.1 * self.env.spec.reward_threshold:
                self.train = False
            elif i > 200:
                self.train = False
            if not self.train:
                print("Episode mean: %f" % episoderewards.mean())
                self.end_count += 1
                if self.end_count > 100:
                    break
            if i % 10 == 0 and i > 1:
                if self.kl_running_avg*self.beta < .1 * self.surr_running_avg:
                    #lower beta
                    self.beta *= 10
                elif self.kl_running_avg*self.beta > 10 * self.surr_running_avg:
                    self.beta *= .1
            if i % 40 == 0 and i > 1:
                self.learning_rate_value *= .1
            if self.train:
                self.vf.fit(paths)
                _,l = self.session.run(
                                [self.train_op,
                                self.proximal_loss],
                                feed_dict=feed)

                l_list = self.session.run(
                                [self.losses],
                                feed_dict=feed)
                l_list = l_list[0]

                stats = {}
                self.kl_running_avg = self.kl_running_avg * .8 + np.abs(l_list[1])*.2
                self.surr_running_avg = self.surr_running_avg * .8 + np.abs(l_list[0])*.2
                numeptotal += len(episoderewards)
                stats["Total number of episodes"] = numeptotal
                stats["Average sum of rewards per episode"] = episoderewards.mean()
                stats["Entropy"] = l_list[2]
                exp = explained_variance(np.array(baseline_n), np.array(returns_n))
                stats["Baseline explained"] = exp
                stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
                stats["KL between old and new distribution"] = l_list[1]
                stats["KL penalty"] = self.beta
                stats["Surrogate loss"] = l_list[0]

                avg_rewards_summ = tf.Summary(value=[tf.Summary.Value(tag="avg_rewards_sum", simple_value= episoderewards.mean())])
                self.train_writer.add_summary(avg_rewards_summ, i)

                kl_summ = tf.Summary(value=[tf.Summary.Value(tag="kl", simple_value=float(l_list[1]))])
                self.train_writer.add_summary(kl_summ, i)

                surr_summ = tf.Summary(value=[tf.Summary.Value(tag="sur", simple_value=float(l_list[0]))])
                self.train_writer.add_summary(surr_summ, i)

                ent_summ = tf.Summary(value=[tf.Summary.Value(tag="ent", simple_value=float(l_list[2]))])
                self.train_writer.add_summary(ent_summ, i)

                beta_summ = tf.Summary(value=[tf.Summary.Value(tag="beta", simple_value=self.beta)])
                self.train_writer.add_summary(beta_summ, i)

                lr_summ = tf.Summary(value=[tf.Summary.Value(tag="lr", simple_value=self.learning_rate_value)])
                self.train_writer.add_summary(lr_summ, i)

                for k, v in stats.items():
                    print(k + ": " + " " * (40 - len(k)) + str(v))
                if l_list[2] != l_list[2]:
                    #NaN
                    exit(-1)

            i += 1
