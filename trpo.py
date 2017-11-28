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
# from external_optimizer import ScipyOptimizerInterface
from tensorflow.contrib.opt.python.training import external_optimizer

def policy_model(input_layer,hidden_layer_sizes=[64,64],task_size=False,output_size=3):
    net = input_layer

    with tf.variable_scope('policy'):
        for i,hidden_size in enumerate(hidden_layer_sizes):
            with tf.variable_scope('h{}'.format(i)):
                W = tf.Variable(tf.truncated_normal([net.get_shape()[1].value,hidden_size])*.01)
                b = tf.Variable(tf.zeros([hidden_size]))
                h = tf.matmul(net,W) + b
                a = tf.nn.tanh(h)
            net = a
        # with tf.variable_scope('h0'):
        #     W = tf.Variable(tf.truncated_normal([net.get_shape()[1].value,hidden_layer_sizes[0]])*.01)
        #     b = tf.Variable(tf.zeros([hidden_layer_sizes[0]]))
        #     h = tf.matmul(net,W) + b
        #     a = tf.nn.tanh(h)
        # net = a
        # with tf.variable_scope('h1'):
        #     W = tf.Variable(tf.truncated_normal([net.get_shape()[1].value,hidden_layer_sizes[0]])*.01)
        #     b = tf.Variable(tf.zeros([hidden_layer_sizes[0]]))
        #     h = tf.matmul(net,W) + b
        #     a = tf.nn.tanh(h)
        # net = a
        with tf.variable_scope('out_mean'):
            W = tf.Variable(tf.zeros([net.get_shape()[1].value,output_size]))
            b = tf.Variable(tf.zeros([output_size]))
            h3 = tf.matmul(net,W) + b
        #fixed std dev
        with tf.variable_scope('out_std'):
            W = tf.Variable(tf.zeros([1,output_size]))
            h3_std = tf.tile(tf.exp(W),tf.stack([tf.shape(h3)[0],1]))
        h3 = tf.reshape(h3,[-1,output_size])
        h3_std = tf.reshape(h3_std,[-1,output_size])
        output = tf.concat(axis=1,values=[h3,h3_std])

    return output

def loglikelihood(actions,action_dist,dims):
    mean_n = tf.reshape(action_dist[:,:dims],[tf.shape(action_dist)[0],dims])
    std_n = (tf.reshape(action_dist[:,dims:],[tf.shape(action_dist)[0],dims]))
    # std_n = tf.reshape(action_dist[:,1,:],[tf.shape(action_dist)[0],tf.shape(action_dist)[2]])
    return -0.5 * tf.reduce_sum(tf.square((actions-mean_n) / std_n),axis=-1) \
                -0.5 * tf.log(2.0*np.pi)*dims - tf.reduce_sum(tf.log(std_n),axis=-1)

def kl_divergence(action_dist_0,action_dist_1,dims):
    #old, new
    mean_0 = tf.reshape(action_dist_0[:,:dims],[tf.shape(action_dist_0)[0],dims])
    std_0 = (tf.reshape(action_dist_0[:,dims:],[tf.shape(action_dist_0)[0],dims]))
    # std_n = tf.reshape(action_dist[:,1,:],[tf.shape(action_dist)[0],tf.shape(action_dist)[2]])
    mean_1 = tf.reshape(action_dist_1[:,:dims],[tf.shape(action_dist_0)[0],dims])
    std_1 = (tf.reshape(action_dist_1[:,dims:],[tf.shape(action_dist_0)[0],dims]))
    numerator = tf.square(mean_0 - mean_1) + tf.square(std_0) - tf.square(std_1)
    denominator = 2 * tf.square(std_1) + 1e-8

    return tf.reduce_sum(
        numerator / denominator + tf.log(std_1) - tf.log(std_0),axis=-1)

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
        "max_pathlength": 250,
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
        self.beta = 1
        self.learning_rate = tf.placeholder(dtype,shape=(),name="learning_rate")
        self.clip = 5
        self.eps = 1e-8
        self.count = None
        self.mean = None
        self.std = None
        self.save_count = 0
        # with subprocess.Popen(["git","rev-parse" ,"HEAD"], stdout=subprocess.PIPE) as proc:
        #     GIT_COMMIT = str(proc.stdout.read())[-8:-3]
        # GIT_COMMIT  = subprocess.check_output(["git", "describe"]).strip()
        GIT_COMMIT  = subprocess.check_output(["git","rev-parse" ,"HEAD"]).strip()[:6]
        # print("GIT", GIT_COMMIT)
        now = datetime.datetime.now()
        self.identifying_string = "{}-{}-{}-rand{}:".format(now.day,now.month,now.year,str(time.time())[-3:]) + GIT_COMMIT
        self.train_writer = tf.summary.FileWriter('./train/'+self.identifying_string,
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
            with tf.variable_scope("policy"):
                action_dist_n, _ = (pt.wrap(self.obs).
                                    fully_connected(32, activation_fn=tf.nn.tanh).
                                    fully_connected(32, activation_fn=tf.nn.tanh).
                                    softmax_classifier(env.action_space.n))
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy')
            self.action_dist_n = action_dist_n
            N = tf.shape(obs)[0]
            p_n = slice_2d(action_dist_n, tf.range(0, N), action)
            oldp_n = slice_2d(oldaction_dist, tf.range(0, N), action)
            ratio_n = p_n / oldp_n
            Nf = tf.cast(N, dtype)
            surr = -tf.reduce_mean(ratio_n * advant)  # Surrogate loss
            kl = tf.reduce_sum(oldaction_dist * tf.log((oldaction_dist + self.eps) / (action_dist_n + self.eps))) / Nf
            ent = tf.reduce_sum(-action_dist_n * tf.log(action_dist_n + self.eps)) / Nf

            grads = tf.gradients(tf.reduce_sum(tf.stop_gradient(action_dist_n) * tf.log((tf.stop_gradient(action_dist_n) + self.eps) / (action_dist_n + self.eps))) / Nf,var_list)

        else:
            self.obs = obs = tf.placeholder(
                dtype, shape=[
                    None, 2 * env.observation_space.shape[0] + env.action_space.shape[0]], name="obs")
            self.dimensions = env.action_space.shape[0]
            self.prev_action = np.zeros((1, env.action_space.shape[0]))
            self.oldaction_dist = oldaction_dist = tf.placeholder(dtype, shape=[None, env.action_space.shape[0]*2], name="oldaction_dist")
            with self.session as sess:
                action_dist_n = policy_model(self.obs,
                                            hidden_layer_sizes=[30,30,15],
                                            output_size=self.dimensions)
            self.action = action = tf.placeholder(dtype, shape=[None,env.action_space.shape[0]], name="action")
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy')
            self.action_dist_n = action_dist_n
            N = tf.shape(obs)[0]
            Nf = tf.cast(N, dtype)
            logp_n = loglikelihood(action,action_dist_n,self.dimensions)
            oldlogp_n = loglikelihood(action,oldaction_dist,self.dimensions)
            surr = -tf.reduce_mean( tf.exp(logp_n - oldlogp_n) * advant)
            kl = tf.reduce_mean(kl_divergence(oldaction_dist,action_dist_n,self.dimensions))
            flipped_kl = tf.reduce_mean(kl_divergence(action_dist_n,oldaction_dist,self.dimensions))
            ent = tf.reduce_mean(tf.reduce_sum(tf.log(action_dist_n[:,self.dimensions:]),1) + .5  *
            np.log(2*np.pi*np.e) * self.dimensions)
            # ent = tf.Variable(0)
            grads = tf.gradients(tf.reduce_mean(kl_divergence(tf.stop_gradient(action_dist_n),action_dist_n,self.dimensions)),var_list)

        # Create neural network.


        self.losses = [surr, kl, ent]
        self.pg = flatgrad(surr,var_list)
        # grads = tf.gradients(kl,var_list)
        self.flat_tangent = tf.placeholder(dtype,shape=[None])
        shapes = []
        for v in var_list:
            shapes.append( var_shape(v) )
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size
        gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        self.fvp = flatgrad(gvp, var_list)
        self.gf = GetFlat(self.session, var_list)
        self.sff = SetFromFlat(self.session, var_list)
        self.proximal_loss = surr + self.beta * kl
        self.learning_rate_value = .01
        self.optimizer = tf.train.GradientDescentOptimizer(1)
        self.sffg = SetFromFlatGrads(self.session,self.optimizer,var_list)

        self.train_op = self.optimizer.minimize(self.proximal_loss)
        self.grads_and_vars = self.optimizer.compute_gradients(self.proximal_loss,var_list)
        print(self.grads_and_vars[0][0].get_shape())
        self.grads_input = tf.placeholder(dtype,shape=[None])
        self.neg_grads_and_vars = [(-gv[0]/100.0,gv[1]) for gv in self.grads_and_vars]

        # self.apply_grads = self.optimizer.apply_gradients(self.grads_input,var_list)
        self.apply_neg_grads = self.optimizer.apply_gradients(self.neg_grads_and_vars)
        self.scipy_optimizer = external_optimizer.ScipyOptimizerInterface(self.proximal_loss,options={'maxiter': 20})
        self.train_op = tf.train.AdamOptimizer(.05,beta1=.1,beta2=.1).minimize(self.proximal_loss) #
        #see if separating them does anything to learning
        self.surr_train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(surr)
        self.kl_train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.beta * kl)
        self.vf = VF(self.session)
        self.kl_running_avg = 0
        self.surr_running_avg = 0
        self.session.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.weights = [tf.reduce_mean(v) for v in tf.trainable_variables() if v.name[:len('policy/h0')]=='policy/h0']

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
            # action = np.random.randn(self.dimensions) * action_dist_n[0,1,:] + action_dist_n[0,0,:]
            # print(action_dist_n.shape)
            action = np.random.randn(self.dimensions) * action_dist_n[0,self.dimensions:] + action_dist_n[0,:self.dimensions]
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
    
    def load_model(self,model_path):
        self.saver.restore(self.session, model_path)

    def play_slow(self):
        ob = self.env.reset()
        agent.prev_action *= 0.0
        agent.prev_obs *= 0.0
        for i in xrange(250):
            time.sleep(.1)
            action, action_dist, ob = agent.act(ob)
            res = env.step(action)
            ob = res[0]
            print('action',action)
            print('reward',res[1])

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
            action_dist_r = np.concatenate([path["action_dists"] for path in paths])
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



            episoderewards = np.array(
                [path["rewards"].sum() for path in paths])

            print("\n********** Iteration %i ************" % i)
            # if self.env.spec.reward_threshold and \
            #     episoderewards.mean() > 1.1 * self.env.spec.reward_threshold:
            #     self.train = False
            if episoderewards.mean() > -.1:
                self.train = False
            elif i > 1000:
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
            # if i % 40 == 0 and i > 1:
            #     self.learning_rate_value *= .1
            if self.train:
                self.vf.fit(paths)

                feed = {self.obs: obs_n,
                        self.action: action_n,
                        self.advant: advant_n,
                        self.oldaction_dist: action_dist_r,
                        self.learning_rate : self.learning_rate_value,
                        }

                if i < 1000:
                    #conj grad style
                    thprev = self.gf()
                    def fisher_vector_product(p):
                        feed[self.flat_tangent] = p
                        return self.session.run(self.fvp, feed) + config.cg_damping * p

                    g = self.session.run(self.pg, feed_dict=feed)
                    print(np.mean(g))
                    stepdir = conjugate_gradient(fisher_vector_product, -g)
                    shs = .5 * stepdir.dot(fisher_vector_product(stepdir))
                    lm = np.sqrt(shs / config.max_kl)
                    print('lm',lm)
                    fullstep = stepdir / lm
                    neggdotstepdir = -g.dot(stepdir)
                    print('expected_improve',neggdotstepdir/lm)
                    def loss_update(th):
                        self.sff(th)
                        return self.session.run(self.losses[:2], feed_dict=feed)
                    def loss_grads(grads):
                        self.sffg(grads)
                        return self.session.run(self.losses[0], feed_dict=feed)
                    print('surr,kl',self.session.run(self.losses[:2], feed_dict=feed))
                    theta = linesearch(loss_update, thprev, fullstep, neggdotstepdir / lm)
                    self.sff(theta)
                    kl_val = self.session.run(
                                    [self.losses[1]],
                                    feed_dict=feed)
                    print(kl_val)
                    if np.mean(kl_val) > .02:
                        print('---------------using old th')
                        self.sff(thprev + fullstep/1000.0)
                else:
                    #sgd style
                #     _,kl_val = self.session.run(
                #                     [self.train_op,self.losses[1]],
                #                     feed_dict=feed)
                # print("kl_val",kl_val)
                # for __ in range(10):
                #
                #     _,kl_val = self.session.run(
                #                     [self.train_op,self.losses[1]],
                #                     feed_dict=feed)
                #     print(kl_val)
                #     if kl_val > 0.01 or kl_val == 0:
                #         break
                #
                    print('scipy proximal loss')
                    self.scipy_optimizer.minimize(self.session,
                            feed_dict=feed)

                ad = self.session.run(
                                [self.action_dist_n],
                                feed_dict=feed)
                ad = ad[0]
                print(ad.shape)
                print(np.mean(ad[:,:],axis=0))#,np.mean(ad[:,1,:],axis=0))
                print(np.std(ad[:,:],axis=0))#,np.std(ad[:,1,:],axis=0))

                w = self.session.run(
                                [self.weights],
                                feed_dict=feed)
                print("weights",w)
                act_mean_summ = tf.Summary(value=[tf.Summary.Value(tag="action_dist_mean_0", simple_value= float(np.mean(ad[:,:],axis=0)[0]))])
                self.train_writer.add_summary(act_mean_summ, i)

                act_std_summ = tf.Summary(value=[tf.Summary.Value(tag="action_dist_std_-1", simple_value= float(np.mean(ad[:,:],axis=0)[-1]))])
                self.train_writer.add_summary(act_std_summ, i)

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
                self.saver.save(self.session,"train/trpo-model-{}-{}.ckpt".format(self.identifying_string,self.save_count))
                self.save_count += 1
                for k, v in stats.items():
                    print(k + ": " + " " * (40 - len(k)) + str(v))
                # if l_list[2] != l_list[2]:
                #     #NaN
                #     exit(-1)

            i += 1
