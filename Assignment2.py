import gym
import numpy as np
import matplotlib.pyplot as plt

MOUNTAIN_CAR_ENV = 'MountainCar-v0'
ENV = gym.make(MOUNTAIN_CAR_ENV).env

ACTIONS = [0, 1, 2]
ACTIONS_NUM = len(ACTIONS)

POLICIES_NUM = 5
EPISODES_NUM = 12000
SAMPLED_EPISODES = 1000
GRAPH_VALUES_NUM = int(EPISODES_NUM / SAMPLED_EPISODES)

EVAL_STEPS_NUM = 200
EVAL_EPISODES_NUM = 100

GAMA = 0.95
DECAY = 0.5


def eval_current_adr(tiles, q_tables):
    s = ENV.reset()
    s_codes = get_tiles_codes(s, tiles)
    success_episodes = 0
    total_steps = 0
    tmp_steps = 0
    failed_episodes = 0

    while True:
        action = get_max_action(s_codes, q_tables)
        s, r, terminated, _ = ENV.step(action)
        s_codes = get_tiles_codes(s, tiles)
        tmp_steps += 1
        if terminated:
            success_episodes += 1
            total_steps += tmp_steps
            s = ENV.reset()
            s_codes = get_tiles_codes(s, tiles)
            tmp_steps = 0
        if tmp_steps == 500:
            failed_episodes += 1
            tmp_steps = 0
            s = ENV.reset()
            s_codes = get_tiles_codes(s, tiles)

        if success_episodes == EVAL_EPISODES_NUM - failed_episodes:
            break

    return total_steps / success_episodes if success_episodes else 300


def create_tiles(num_tiles, grids, offsets, feature_ranges):
    features_num = len(feature_ranges)
    tiles = []
    for i in range(num_tiles):
        curr_grid = grids[i]
        curr_offsets = offsets[i]

        tile = []
        for curr_feature in range(features_num):
            curr_feature_range = feature_ranges[curr_feature]
            curr_feature_bins = curr_grid[curr_feature]
            curr_feature_offset = curr_offsets[curr_feature]
            feat_tiling = np.linspace(curr_feature_range[0],
                                      curr_feature_range[1],
                                      curr_feature_bins + 1)[1:-1] + curr_feature_offset
            tile.append(feat_tiling)
        tiles.append(tile)
    return np.array(tiles)


def get_max_action(s_codes, q_tables):
    return np.argmax([get_state_action_value(s_codes, a, q_tables) for a in ACTIONS])


def get_max_action_value(s_codes, q_tables):
    return np.max([get_state_action_value(s_codes, a, q_tables) for a in ACTIONS])


def choose_action_epsilon_greedy(s_codes, q_tables, epsilon):
    if np.random.uniform() < epsilon:
        a = ENV.action_space.sample()
    else:
        a = get_max_action(s_codes, q_tables)
    return a


def get_tiles_codes(s, tiles):
    codes = []
    for tile in tiles:
        code = []
        for feature in range(len(s)):
            curr_coding = np.digitize(s[feature], tile[feature])
            code.append(curr_coding)
        codes.append(code)
    return np.array(codes)


def get_state_action_value(s_codes, a, q_tables):
    tiles_num = len(q_tables)
    val = 0
    for tile_number in range(tiles_num):
        curr_q_table = q_tables[tile_number]
        val += curr_q_table[tuple(s_codes[tile_number]) + (a,)]
    return val / tiles_num


def update_q_tables(s_codes, a, q_tables, update):
    tiles_num = len(q_tables)
    for tile_number in range(tiles_num):
        curr_q_table = q_tables[tile_number]
        curr_codes = tuple(s_codes[tile_number]) + (a,)
        # delta = q_target - curr_q_table[curr_codes]
        curr_q_table[curr_codes] += update


def tile_q_learn(alpha):
    # velocity range -0.07 to 0.07
    # position range -1.2 to 0.6
    # there are 2 features: height, velocity
    feature_range = [[-1.2, 0.6], [-0.07, 0.07]]

    # defines how many bins each feature gets in each tile i.e. what are each grid sizes
    grids = [[15, 15], [15, 15]]
    tiles_num = len(grids)

    offsets = [[0, 0], [0.2, 0.015]]

    tiles = create_tiles(tiles_num, grids, offsets, feature_range)

    # create q_table for each grid in its dimensions * ACTION_NUM(=3)
    q_tables = [np.zeros(tuple(grid) + (ACTIONS_NUM,)) for grid in grids]

    epsilon = 0.1

    results = np.zeros((GRAPH_VALUES_NUM, 1))

    for episode in range(EPISODES_NUM):
        s = ENV.reset()
        s_codes = get_tiles_codes(s, tiles)

        for step in range(EVAL_STEPS_NUM):

            a = choose_action_epsilon_greedy(s_codes, q_tables, epsilon)
            next_s, r, terminated, _ = ENV.step(a)

            next_s_codes = get_tiles_codes(next_s, tiles)

            update = alpha * ((GAMA * get_max_action_value(next_s_codes, q_tables) + r) -
                              get_state_action_value(s_codes, a, q_tables))

            update_q_tables(s_codes, a, q_tables, update)

            if terminated:
                break

            s_codes = next_s_codes

        if (episode + 1) % SAMPLED_EPISODES == 0:
            alpha /= 1 + DECAY * (episode / SAMPLED_EPISODES)
            sample_index = int((episode + 1) / SAMPLED_EPISODES) - 1
            results[sample_index] = eval_current_adr(tiles, q_tables)
            print(f'episode {episode}, spe: {results[sample_index]}')

    return results


def plot_graph(final_results):
    plt.plot([i * SAMPLED_EPISODES for i in range(1, GRAPH_VALUES_NUM + 1)],
             [np.average(final_results[i]) for i in range(GRAPH_VALUES_NUM)])
    plt.xlabel('episodes')
    plt.ylabel('steps per trial')
    plt.show()


def main():
    alpha = float(input('Insert alpha: '))
    final_results = np.zeros((GRAPH_VALUES_NUM, POLICIES_NUM))
    for policy in range(POLICIES_NUM):
        print(f'Policy {policy}:')
        curr_adr = tile_q_learn(alpha)
        for value in range(GRAPH_VALUES_NUM):
            final_results[value][policy] += curr_adr[value]

    plot_graph(final_results)


if __name__ == '__main__':
    main()
