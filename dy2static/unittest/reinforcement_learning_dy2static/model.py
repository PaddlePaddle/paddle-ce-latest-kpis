# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
import math
import argparse
import numpy as np
from time import time

import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph.nn as nn
from paddle.fluid.dygraph import to_variable, Layer
from paddle.jit import to_static, ProgramTranslator

SEED = 2020
program_translator = ProgramTranslator()


class Policy(Layer):
    def __init__(self):
        super(Policy, self).__init__()

        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)
        self.dropout_ratio = 0.6

        self.saved_log_probs = []
        self.rewards = []

    @to_static
    def forward(self, x):
        x = fluid.layers.reshape(x, shape=[1, 4])
        x = self.affine1(x)
        x = fluid.layers.dropout(x, self.dropout_ratio)
        x = fluid.layers.relu(x)
        action_scores = self.affine2(x)

        log_prob = fluid.layers.softmax(action_scores, axis=1)

        return log_prob


def parse_args():
    parser = argparse.ArgumentParser("reinforcement learning model benchmark.")
    # parser.add_argument(
    #     '--to_static', type=bool, default=True, help='whether to train model in static mode.')
    parser.add_argument(
        '--gamma', type=float, default=0.99, help='gamma value')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='The number for log interval.')
    parser.add_argument(
        '--pass_num', type=int, default=360, help='The number of passes.')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def train(args, to_static):
    program_translator.enable(to_static)

    # init environment
    env = gym.make('CartPole-v0')
    env.seed(SEED)
    paddle.seed(SEED)
    paddle.framework.random._manual_program_seed(SEED)
    local_random = np.random.RandomState(SEED)

    # set device
    device = 'gpu:0' if fluid.is_compiled_with_cuda(
    ) and args.device == 'GPU' else 'cpu'
    paddle.set_device(device)

    # create model
    policy = Policy()
    eps = np.finfo(np.float32).eps.item()
    optimizer = fluid.optimizer.AdamaxOptimizer(
        learning_rate=1e-2, parameter_list=policy.parameters())

    def get_mean_and_std(values=[]):
        n = 0.
        s = 0.
        for val in values:
            s += val
            n += 1
        mean = s / n

        std = 0.
        for val in values:
            std += (val - mean) * (val - mean)
        std /= n
        std = math.sqrt(std)

        return mean, std

    def sample_action(probs):
        sample = local_random.random_sample()
        idx = 0

        while idx < len(probs) and sample > probs[idx]:
            sample -= probs[idx]
            idx += 1
        mask = [0.] * len(probs)
        mask[idx] = 1.

        return idx, np.array([mask]).astype("float32")

    def choose_best_action(probs):
        idx = 0 if probs[0] > probs[1] else 1
        mask = [1., 0.] if idx == 0 else [0., 1.]

        return idx, np.array([mask]).astype("float32")

    def select_action(state):
        state = to_variable(state)
        state.stop_gradient = True
        loss_probs = policy(state)

        probs = loss_probs.numpy()

        action, _mask = sample_action(probs[0])
        mask = to_variable(_mask)
        mask.stop_gradient = True

        loss_probs = fluid.layers.log(loss_probs)
        loss_probs = fluid.layers.elementwise_mul(loss_probs, mask)
        loss_probs = fluid.layers.reduce_sum(loss_probs, dim=-1)

        policy.saved_log_probs.append(loss_probs)
        return action, loss_probs

    def finish_episode():
        R = 0
        policy_loss = []
        returns = []
        for r in policy.rewards[::-1]:
            R = r + args.gamma * R
            returns.insert(0, R)

        mean, std = get_mean_and_std(returns)

        returns = np.array(returns).astype("float32")
        returns = (returns - mean) / (std + eps)

        # calculate policy loss of each step.
        for log_prob, R in zip(policy.saved_log_probs, returns):
            log_prob_numpy = log_prob.numpy()

            R_numpy = np.ones_like(log_prob_numpy).astype("float32")
            _R = -1 * R * R_numpy
            _R = to_variable(_R)
            _R.stop_gradient = True
            cur_loss = fluid.layers.elementwise_mul(_R, log_prob)
            policy_loss.append(cur_loss)

        policy_loss = fluid.layers.concat(policy_loss)
        policy_loss = fluid.layers.reduce_sum(policy_loss)

        policy_loss.backward()
        optimizer.minimize(policy_loss)
        policy.clear_gradients()

        del policy.rewards[:]
        del policy.saved_log_probs[:]

        return returns

    loss_data = []
    running_reward = 10
    for i_episode in range(args.pass_num):
        avg_cost_time = []
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):
            start_t = time()

            state = np.array(state).astype("float32")
            action, loss = select_action(state)
            state, reward, done, _ = env.step(action)

            end_t = time()
            avg_cost_time.append((end_t - start_t) * 1000)  # ms

            # log loss_probs
            loss_data.append(loss.numpy()[0])
            policy.rewards.append(reward)
            ep_reward += reward

            # training finished
            if done:
                break

        # sum loss and apply optimization
        returns = finish_episode()

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        if i_episode % args.log_interval == 0:
            print(
                'ToStatic = {},\tPass = {},\tLast reward = {:.2f},\tAverage reward = {:.2f},\tloss_probs = {:.6f},\tElapse(ms) = {:.3f}'.
                format(to_static, i_episode, ep_reward, running_reward,
                       loss.numpy()[0], np.mean(avg_cost_time)))

    return np.array(loss_data)


def run_benchmark(args):
    # train in dygraph mode
    train(args, to_static=False)

    # train in static mode
    train(args, to_static=True)


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    # train model
    run_benchmark(args)
