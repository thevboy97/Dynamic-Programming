import os
import numpy as np
import gym
import gym_minigrid
import pickle
import matplotlib.pyplot as plt
import imageio
import random

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door


def step_cost(action):
    # You should implement the stage cost by yourself
    # Feel free to use it or not
    # ************************************************
    # the cost of action
    # if action in [MF, PK, UD]:
    #     return 1
    # if action in [TL, TR]:
    #     return 2
    return 1


def step(env, action):
    '''
    Take Action
    ----------------------------------
    actions:
        0 # Move forward (MF)
        1 # Turn left (TL)
        2 # Turn right (TR)
        3 # Pickup the key (PK)
        4 # Unlock the door (UD)
    '''
    actions = {
        0: env.actions.forward,
        1: env.actions.left,
        2: env.actions.right,
        3: env.actions.pickup,
        4: env.actions.toggle
    }

    _, _, done, _ = env.step(actions[action])
    return step_cost(action), done


def generate_random_env(seed, task):
    ''' 
    Generate a random environment for testing
    -----------------------------------------
    seed:
        A Positive Integer,
        the same seed always produces the same environment
    task:
        'MiniGrid-DoorKey-5x5-v0'
        'MiniGrid-DoorKey-6x6-v0'
        'MiniGrid-DoorKey-8x8-v0'
    '''
    if seed < 0:
        seed = np.random.randint(50)
    env = gym.make(task)
    env.seed(seed)
    env.reset()
    return env


def load_env(path):
    '''
    Load Environments
    ---------------------------------------------
    Returns:
        gym-environment, info
    '''
    with open(path, 'rb') as f:
        env = pickle.load(f)

    info = {
        'height': env.height,
        'width': env.width,
        'init_agent_pos': env.agent_pos,
        'init_agent_dir': env.dir_vec
    }

    for i in range(env.height):
        for j in range(env.width):
            if isinstance(env.grid.get(j, i),
                          gym_minigrid.minigrid.Key):
                info['key_pos'] = np.array([j, i])
            elif isinstance(env.grid.get(j, i),
                            gym_minigrid.minigrid.Door):
                info['door_pos'] = np.array([j, i])
            elif isinstance(env.grid.get(j, i),
                            gym_minigrid.minigrid.Goal):
                info['goal_pos'] = np.array([j, i])

    return env, info


def load_random_env(env_folder):
    '''
    Load a random DoorKey environment
    ---------------------------------------------
    Returns:
        gym-environment, info
    '''
    env_list = [os.path.join(env_folder, env_file)
                for env_file in os.listdir(env_folder)]
    env_path = random.choice(env_list)
    with open(env_path, 'rb') as f:
        env = pickle.load(f)

    info = {
        'height': env.height,
        'width': env.width,
        'init_agent_pos': env.agent_pos,
        'init_agent_dir': env.dir_vec,
        'door_pos': [],
        'door_open': [],
    }

    for i in range(env.height):
        for j in range(env.width):
            if isinstance(env.grid.get(j, i),
                          gym_minigrid.minigrid.Key):
                info['key_pos'] = np.array([j, i])
            elif isinstance(env.grid.get(j, i),
                            gym_minigrid.minigrid.Door):
                info['door_pos'].append(np.array([j, i]))
                if env.grid.get(j, i).is_open:
                    info['door_open'].append(True)
                else:
                    info['door_open'].append(False)
            elif isinstance(env.grid.get(j, i),
                            gym_minigrid.minigrid.Goal):
                info['goal_pos'] = np.array([j, i])

    return env, info, env_path


def save_env(env, path):
    with open(path, 'wb') as f:
        pickle.dump(env, f)


def plot_env(env):
    '''
    Plot current environment
    ----------------------------------
    '''
    img = env.render('rgb_array', tile_size=32)
    plt.figure()
    plt.imshow(img)
    plt.show()


def draw_gif_from_seq(seq, env, path='/Users/vb/Desktop/ece276b/ECE276B_PR1/starter_code/gif/doorkey.gif'):
    '''
    Save gif with a given action sequence
    ----------------------------------------
    seq:
        Action sequence, e.g [0,0,0,0] or [MF, MF, MF, MF]

    env:
        The doorkey environment
    '''
    with imageio.get_writer(path, mode='I', duration=0.8) as writer:
        img = env.render('rgb_array', tile_size=32)
        writer.append_data(img)
        for act in seq:
            img = env.render('rgb_array', tile_size=32)
            step(env, act)
            writer.append_data(img)
    print('GIF is written to {}'.format(path))
    return


def child(env, info, cell):
    door = env.grid.get(info['door_pos'][0], info['door_pos'][1])
    col, row = cell
    n = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    l = []
    for i in n:
        nc, nr = col + i[0], row + i[1]
        loc = type(env.grid.get(nr, nc))
        if loc is not gym_minigrid.minigrid.Wall and env.width > nc >= 0 and env.height > nr >= 0:
            if loc is gym_minigrid.minigrid.Door:
                if door.is_locked:
                    if env.carrying is not None:
                        l.append((nc, nr))
                        continue
                    else:
                        continue
            l.append((nc, nr))
    return l


def child_door(env, cell):
    col, row = cell
    n = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    l = []
    for i in n:
        nc, nr = col + i[0], row + i[1]
        loc = type(env.grid.get(nr, nc))
        if loc is not gym_minigrid.minigrid.Wall and env.width > nc >= 0 and env.height > nr >= 0:
            l.append((nc, nr))
    return l


def move(env, ini, fin):
    agent_dir = env.dir_vec
    diff = np.array(fin[::-1]) - np.array(ini[::-1])
    # front
    if (agent_dir == diff).all():
        step(env, MF)
        action = [MF]
    # back
    if (agent_dir == -diff).all():
        step(env, TR)
        step(env, TR)
        step(env, MF)
        action = [TR, TR, MF]
    # right
    if ((agent_dir == [0, -1]).all() and (diff == [1, 0]).all()) or ((agent_dir == [1, 0]).all() and (diff == [0, 1]).all()) \
            or ((agent_dir == [0, 1]).all() and (diff == [-1, 0]).all()) or ((agent_dir == [-1, 0]).all() and (diff == [0, -1]).all()):
        step(env, TR)
        step(env, MF)
        action = [TR, MF]
    # left
    if ((agent_dir == [0, -1]).all() and (diff == [-1, 0]).all()) or ((agent_dir == [-1, 0]).all() and (diff == [0, 1]).all()) \
            or ((agent_dir == [0, 1]).all() and (diff == [1, 0]).all()) or ((agent_dir == [1, 0]).all() and (diff == [0, -1]).all()):
        step(env, TL)
        step(env, MF)
        action = [TL, MF]
    return action


def change_dir(env, init_dir):
    agent_dir = env.dir_vec
    # right
    if ((agent_dir == [0, -1]).all() and (init_dir == [1, 0]).all()) or ((agent_dir == [1, 0]).all() and (init_dir == [0, 1]).all()) \
            or ((agent_dir == [0, 1]).all() and (init_dir == [-1, 0]).all()) or ((agent_dir == [-1, 0]).all() and (init_dir == [0, -1]).all()):
        step(env, TR)
    # left
    if ((agent_dir == [0, -1]).all() and (init_dir == [-1, 0]).all()) or ((agent_dir == [-1, 0]).all() and (init_dir == [0, 1]).all()) \
            or ((agent_dir == [0, 1]).all() and (init_dir == [1, 0]).all()) or ((agent_dir == [1, 0]).all() and (init_dir == [0, -1]).all()):
        step(env, TL)
    # back
    if (agent_dir == -init_dir).all():
        step(env, TR)
        step(env, TR)
    pass


def dpa(env, info, st, en):
    curr_dir = env.dir_vec
    s = tuple(st)
    t = tuple(en)
    start = s[::-1]
    end = t[::-1]
    open = [start]
    h = env.height
    w = env.width
    g = np.ones((h, w)) * np.inf
    g[start] = 0
    parent = {}
    iter = 0
    while len(open):
        iter += 1
        i = open.pop()
        if (en == info['key_pos']).all():
            tester = child(env, info, i)
        else:
            tester = child_door(env, i)
        for j in tester:
            action = move(env, i, j)
            cost = sum([step_cost(k) for k in action])
            check = g[i] + cost
            if check < g[j] and check < g[end]:
                g[j] = check
                parent[j] = i
                if j != end:
                    open.append(j)

    loc = [end]
    temp = end
    while temp != start:
        temp = parent[temp]
        loc.append(temp)
    loc_seq = loc[::-1]
    env.agent_pos = st
    change_dir(env, curr_dir)
    est_seq = []
    for i in range(len(loc_seq)-1):
        state = move(env, loc_seq[i], loc_seq[i+1])
        est_seq.extend(state)
    del est_seq[-1]
    # print('parent', parent)
    # print('label', g)
    # print('loc', loc_seq)
    return est_seq


def dpa_dir(env, info):

    env.agent_pos = info['init_agent_pos']
    change_dir(env, info['init_agent_dir'])

    curr_dir = env.dir_vec
    s = tuple(env.agent_pos)
    t = tuple(info['goal_pos'])
    start = s[::-1]
    end = t[::-1]
    open = [start]
    h = env.height
    w = env.width
    g = np.ones((h, w)) * np.inf
    g[start] = 0
    parent = {}
    iter = 0
    while len(open):
        iter += 1
        i = open.pop()
        tester = child_door(env, i)
        for j in tester:
            action = move(env, i, j)
            cost = sum([step_cost(k) for k in action])
            check = g[i] + cost
            if check < g[j] and check < g[end]:
                g[j] = check
                parent[j] = i
                if j != end:
                    open.append(j)
    flag = 1
    loc = [end]
    temp = end
    while temp != start:
        temp = parent[temp]
        loc.append(temp)
    if tuple(info['door_pos'][::-1]) in loc:
        flag = 0
    loc_seq = loc[::-1]
    env.agent_pos = info['init_agent_pos']
    change_dir(env, curr_dir)
    est_seq = []
    for i in range(len(loc_seq)-1):
        state = move(env, loc_seq[i], loc_seq[i+1])
        est_seq.extend(state)
    return est_seq, flag
