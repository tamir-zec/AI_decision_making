import gym
import numpy as np
import matplotlib.pyplot as plt

MOUNTAIN_CAR_ENV = 'MountainCarContinuous-v0'
ENV = gym.make(MOUNTAIN_CAR_ENV).env

POLICIES_NUM = 10
EPISODES_NUM = 300
SAMPLED_EPISODES = 30
GRAPH_VALUES_NUM = int(EPISODES_NUM / SAMPLED_EPISODES)

EVAL_STEPS_NUM = 500
EVAL_EPISODES_NUM = 20

GAMA = 1

ALPHA_DECAY = 0.01
SIGMA_DECAY = 0.001


def eval_current_adr(theta):
    s = ENV.reset()
    s_features = feautrize_state(s)
    success_episodes = 0
    total_steps = 0
    tmp_steps = 0
    failed_episodes = 0

    while True:
        action = np.clip(s_features @ theta, -1, 1)
        s, r, terminated, _ = ENV.step(action)
        s_features = feautrize_state(s)
        tmp_steps += 1
        if terminated:
            success_episodes += 1
            total_steps += tmp_steps
            s = ENV.reset()
            s_features = feautrize_state(s)
            tmp_steps = 0
        if tmp_steps == 500:
            failed_episodes += 1
            tmp_steps = 0
            s = ENV.reset()
            s_features = feautrize_state(s)

        if success_episodes == EVAL_EPISODES_NUM - failed_episodes:
            break

    return total_steps / success_episodes if success_episodes else 500


def sample_action(s_features, theta, sigma):
    mu = s_features @ theta
    a = np.random.normal(mu, sigma)
    action = np.clip(a, -1, 1)
    return action


def get_score_value(s_features, a, theta, sigma):
    return ((a - s_features @ theta) * s_features) / (sigma ** 2)


def feautrize_state(s):
    loc, velocity = s
    state = np.zeros((1, 5))
    state[0, 0] = 1 if velocity > 0 else 0
    state[0, 1] = 1 if velocity < 0 else 0
    state[0, 2] = 1 if velocity == 0 else 0
    state[0, 3] = 1 if loc * velocity > 0 else 0
    state[0, 4] = 1 if loc * velocity < 0 else 0
    return state


def featurize_state_action(s, a):
    state = np.zeros((1, 3))
    state[0, 0] = s[0]
    state[0, 1] = s[1]
    state[0, 2] = a
    return state


def actor_critic(alpha, beta):
    # position range -1.2 to 0.6
    # velocity range -0.07 to 0.07

    w = np.zeros((3, 1))
    theta = np.zeros((5, 1))

    results = np.zeros((GRAPH_VALUES_NUM, 1))

    sigma = 10

    for episode in range(EPISODES_NUM):
        s = ENV.reset()
        s_features = feautrize_state(s)
        a = sample_action(s_features, theta, sigma)

        for step in range(EVAL_STEPS_NUM):

            s_features = feautrize_state(s)
            s_a_features = featurize_state_action(s, a)

            next_s, r, terminated, _ = ENV.step(a)
            next_s_features = feautrize_state(next_s)

            next_a = sample_action(next_s_features, theta, sigma)
            next_s_a_features = featurize_state_action(next_s, next_a)

            val_func = s_a_features @ w
            next_val_func = next_s_a_features @ w

            score_val = get_score_value(s_features, a, theta, sigma)
            delta = r + GAMA * next_val_func - val_func

            theta += np.transpose(alpha * score_val * val_func)
            if np.max(np.abs(theta)) > 10000:
                theta /= 100
            w += np.transpose(beta * delta * s_a_features)

            s = next_s
            a = next_a

            if terminated:
                break

        if episode % 10 == 0:
            alpha /= 1 + ALPHA_DECAY * (episode / 10)
            sigma /= 1 + SIGMA_DECAY * (episode / 10)

        if (episode + 1) % SAMPLED_EPISODES == 0:
            sample_index = int((episode + 1) / SAMPLED_EPISODES) - 1
            results[sample_index] = eval_current_adr(theta)
            # print(f'episode {episode}, spe: {results[sample_index]}')

    return results


def plot_graph(final_results):
    plt.plot([i * SAMPLED_EPISODES for i in range(1, GRAPH_VALUES_NUM + 1)],
             [np.average(final_results[i]) for i in range(GRAPH_VALUES_NUM)])
    plt.xlabel('episodes')
    plt.ylabel('steps per trial')
    plt.show()


def main():
    alpha = float(input('Insert alpha: '))
    beta = float(input('Insert beta: '))
    final_results = np.zeros((GRAPH_VALUES_NUM, POLICIES_NUM))
    for policy in range(POLICIES_NUM):
        print(f'Policy {policy + 1}')
        curr_adr = actor_critic(alpha, beta)
        for value in range(GRAPH_VALUES_NUM):
            final_results[value][policy] += curr_adr[value]

    plot_graph(final_results)


if __name__ == '__main__':
    main()
