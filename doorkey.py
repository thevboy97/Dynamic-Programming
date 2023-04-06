import numpy as np
import gym
from utils import *
from example import example_use_of_gym_env

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door


def doorkey_problem(env, info):
    '''
    You are required to find the optimal path in
        doorkey-5x5-normal.env
        doorkey-6x6-normal.env
        doorkey-8x8-normal.env

        doorkey-6x6-direct.env
        doorkey-8x8-direct.env

        doorkey-6x6-shortcut.env
        doorkey-8x8-shortcut.env

    Feel Free to modify this fuction
    '''
    start_to_key = dpa(env, info, info['init_agent_pos'], info['key_pos'])
    step(env, PK)
    key_to_door = dpa(env, info, env.agent_pos, info['door_pos'])
    step(env, UD)
    door_to_goal = dpa(env, info, env.agent_pos, info['goal_pos'])
    step(env, MF)
    est_seq = start_to_key + [PK] + key_to_door + [UD] + door_to_goal + [MF]
    direct_seq, flag = dpa_dir(env, info)
    if flag:
        if len(est_seq) < len(direct_seq):
            best_seq = est_seq
        else:
            best_seq = direct_seq
    else:
        best_seq = est_seq
    print('best sequence', best_seq)
    return best_seq


def partA():
    env_path = '/Users/vb/Desktop/ece276b/ECE276B_PR1/starter_code/envs/doorkey-8x8-normal.env'
    env, info = load_env(env_path)  # load an environment
    seq = doorkey_problem(env, info)  # find the optimal action sequence
    draw_gif_from_seq(seq, load_env(env_path)[0])  # draw a GIF & save


def partB():
    env_folder = './envs/random_envs'
    env, info, env_path = load_random_env(env_folder)


if __name__ == '__main__':
    # example_use_of_gym_env()
    partA()
    # partB()
