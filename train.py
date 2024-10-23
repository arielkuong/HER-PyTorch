import numpy as np
import gym
import gym_customized
import os, sys
from arguments import get_args
from mpi4py import MPI
from subprocess import CalledProcessError
from ddpg_agent import ddpg_agent
# from ddpg_agent_wo_norm import ddpg_agent
# from ddpg_agent_wo_norm_record import ddpg_agent
import random
import torch

from gym.envs.registration import register

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""
def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    #params['reward_type'] = env._kwargs.reward_type
    print('Env observation dimension: {}'.format(params['obs']))
    print('Env goal dimension: {}'.format(params['goal']))
    print('Env action dimension: {}'.format(params['action']))
    print('Env max action value: {}'.format(params['action_max']))
    print('Env max timestep value: {}'.format(params['max_timesteps']))
    return params

def launch(args):
    # create the ddpg_agent
    env = gym.make(args.env_name)
    # env = gym_customized.make(args.env_name)
    # set random seeds for reproduce
    #env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    env.seed(args.seed)
    #random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed)
    #np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed)
    #torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed)
    if args.cuda:
        #torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
        torch.cuda.manual_seed(args.seed)
    # get the environment parameters
    env_params = get_env_params(env)

    print('Run training with seed {}'.format(args.seed))
    # create the ddpg agent to interact with the environment
    ddpg_trainer = ddpg_agent(args, env, env_params)
    ddpg_trainer.learn()

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
