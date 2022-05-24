import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from replay_buffer import ReplayBuffer
from policy_critic import Policy, Critic

def select_action(state, mode='policy'):
    if mode == 'random':
        return env.action_space.sample()
    state = torch.from_numpy(state).float()
    probs = model(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item()

def compute_bellman_loss(batch, q, qtarget, pi):
    o = batch['obs']
    a = batch['act']
    r = batch['reward']
    no = batch['next_obs']
    na = pi.sample_actions(no)
    # TODO: Deal with terminals
    target = r + discount*qtarget(no, na)
    bellman_loss = torch.mean((q(o, a) - target)**2)
    return bellman_loss

def compute_actor_loss(batch, q, pi):
    o = batch['obs']
    a = batch['act']
    pi_dist = Categorical(pi(o))
    lp = pi_dist.log_prob(a)
    qv = q(o, a).detach()
    actor_loss = -torch.mean(lp*qv)
    return actor_loss

def compute_losses(q, qtarget, pi, batch_size):
    batch = replay_buffer.sample_batch(batch_size)
    bellman_loss = compute_bellman_loss(batch, q, qtarget, pi)
    actor_loss = compute_actor_loss(batch, q, pi)
    return bellman_loss, actor_loss

def train_step(q, qtarget, pi, batch_size, q_optimizer, pi_optimizer):
    bellman_loss, actor_loss = compute_losses(q, qtarget, pi, batch_size)
    
    # Critic update
    q_optimizer.zero_grad()
    bellman_loss.backward()
    q_optimizer.step()

    # Actor update
    pi_optimizer.zero_grad()
    actor_loss.backward()
    pi_optimizer.step()

def qtarget_flip(q, qtarget, tau):
    for param, target_param in zip(q.parameters(), qtarget.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def evaluate(num_iters):
    # Pretrain data collection
    returns = []

    for traj_num in range(num_iters):
        rewards = []
        state = env.reset()
        for t in range(pretrain_data_steps):
            action = select_action(state)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            state = next_state.copy()
            if done:
                break
        rewards = np.asarray(rewards)
        returns.append(rewards.sum())
    returns = np.asarray(returns)
    return returns


# Hyperparams
discount = 0.99
num_step_before_eval = 1000
num_step_before_train = 1000
train_freq = 1
soft_update_freq = 1
tau = 0.05
pretrain_data_steps = 1000
num_iters_overall = 10000
rb_capacity = 10000

# Parameters and bookkeeping
env = gym.make('CartPole-v1')
obs_shape = env.observation_space.shape[0]
act_shape = env.action_space.n

# Replay buffer and model definitions
buffer = ReplayBuffer(obs_shape, action_shape, rb_capacity, device)
actor = Policy()
critic = Critic()

# Optimizers
q_optimizer = optim.Adam(critic.parameters())
pi_optimizer = optim.Adam(actor.parameters())

# Pretrain data collection
state = env.reset()
for t in range(pretrain_data_steps):
    action = select_action(state, 'random')
    next_state, reward, done, _ = env.step(action)
    replay_buffer.add(state, action, reward, next_state, done)
    state = next_state.copy()
    if done:
        state = env.reset()

# Run training and eval
for eval_iters in range(num_iters_overall):
    state = env.reset()
    for t in range(num_step_before_eval):
        action = select_action(state)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state.copy()
        
        if done:
            state = env.reset()
        
        if replay_buffer.size > num_step_before_train:
            if t % train_freq == 0:
                train_step(q, qtarget, pi, batch_size, q_optimizer, pi_optimizer)

            if t % soft_update_freq == 0:
                qtarget_flip(q, qtarget, tau)

    # Evaluation
    eval_returns = evaluate(num_iters)
    print("Returns at iter %d are %.8f"%(eval_iters, eval_returns.mean()))