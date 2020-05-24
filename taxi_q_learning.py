import argparse

import gym
import numpy as np
import matplotlib.pyplot as plt

TAXI_ENV = 'Taxi-v3'
ENV = gym.make(TAXI_ENV).env
Q_TABLE = np.zeros([ENV.observation_space.n, ENV.action_space.n])

# alpha best value around 0.3

NUM_OF_POLICIES = 10
NUM_TOTAL_STEPS = 100000
NUM_OF_SAMPLED_STEPS = 1000
NUM_OF_GRAPH_VALUES = int(NUM_TOTAL_STEPS / NUM_OF_SAMPLED_STEPS)
NUM_OF_EVAL_STEPS = 100
GAMA = 0.95
WALL_PENALTY = -500

DEBUG_INFO = False


def plot_adr_graph(final_adr_statistics):
    plt.errorbar([i * NUM_OF_SAMPLED_STEPS for i in range(1, NUM_OF_GRAPH_VALUES + 1)],
                 [np.average(final_adr_statistics[i]) for i in range(NUM_OF_GRAPH_VALUES)],
                 [np.var(final_adr_statistics[i]) for i in range(NUM_OF_GRAPH_VALUES)],
                 linestyle='None',
                 marker='^',
                 capsize=3)
    plt.show()


def plot_episodes_graph(final_episodes_statistics):
    plt.plot([i * NUM_OF_SAMPLED_STEPS for i in range(1, NUM_OF_GRAPH_VALUES + 1)],
             [np.average(final_episodes_statistics[i]) for i in range(NUM_OF_GRAPH_VALUES)])
    plt.show()


def eval_current_adr():
    s = ENV.reset()
    reward_accumulated = 0
    for i in range(NUM_OF_EVAL_STEPS):
        action = np.argmax((Q_TABLE[s]))
        s, r, terminated, _ = ENV.step(action)
        reward_accumulated += ((GAMA ** i) * r)
        if terminated:
            s = ENV.reset()

    return reward_accumulated / NUM_OF_EVAL_STEPS


def q_learn(alpha, policy_index):
    s = ENV.reset()
    global Q_TABLE
    Q_TABLE *= 0

    adr_statistics = np.zeros((NUM_OF_GRAPH_VALUES, 1))
    episodes_statistics = np.zeros((NUM_OF_GRAPH_VALUES, 1))
    total_episodes = 0
    last_phase_episodes = 0
    epsilon = 0.1
    terminated = False

    for i in range(NUM_TOTAL_STEPS):

        if terminated:
            total_episodes += 1
            s = ENV.reset()

        if np.random.uniform() < epsilon:
            action = ENV.action_space.sample()
        else:
            max_indices = (np.where(Q_TABLE[s] == max(Q_TABLE[s])))[0]
            action = np.random.choice(max_indices)

        next_s, reward, terminated, _ = ENV.step(action)
        curr_sq_value = Q_TABLE[s][action]

        if s == next_s:
            Q_TABLE[s][action] = WALL_PENALTY
        else:
            Q_TABLE[s][action] = \
                curr_sq_value + alpha * ((GAMA * max(Q_TABLE[next_s])) + reward - curr_sq_value)
        s = next_s

        if (i + 1) % NUM_OF_SAMPLED_STEPS == 0:
            sample_index = int((i + 1) / NUM_OF_SAMPLED_STEPS) - 1
            current_sample_episodes = total_episodes - last_phase_episodes
            adr_statistics[sample_index] = eval_current_adr()
            if current_sample_episodes != 0:
                episodes_statistics[sample_index] = NUM_OF_SAMPLED_STEPS / current_sample_episodes
                last_phase_episodes = total_episodes
            else:
                episodes_statistics[sample_index] = NUM_OF_SAMPLED_STEPS

    if DEBUG_INFO:
        print('Policy {} ran {} episodes'.format(policy_index + 1, total_episodes))

    return adr_statistics, episodes_statistics


def main():
    alpha = float(input('Insert alpha: '))
    ada_final_statistics = np.zeros((NUM_OF_GRAPH_VALUES, NUM_OF_POLICIES))
    episodes_final_statistics = np.zeros((NUM_OF_GRAPH_VALUES, NUM_OF_POLICIES))

    for j in range(NUM_OF_POLICIES):
        curr_adr, curr_episodes = q_learn(alpha, j)
        for i in range(NUM_OF_GRAPH_VALUES):
            ada_final_statistics[i][j] += curr_adr[i]
            episodes_final_statistics[i][j] += curr_episodes[i]

    if DEBUG_INFO:
        print('Final ADR Average: {}'.format(np.average(ada_final_statistics[-1])))
        print('Final Steps per Episode Average: {}'.format(np.average(episodes_final_statistics[-1])))

    plot_adr_graph(ada_final_statistics)
    plot_episodes_graph(episodes_final_statistics)
    ENV.close()


if __name__ == '__main__':
    main()
