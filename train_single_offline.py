import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torchkit.pytorch_utils import set_gpu_mode
import utils.config_utils as config_utl
from utils import helpers as utl, offline_utils as off_utl
from offline_rl_config import args_gridworld_block, args_cheetah_vel, args_ant_dir, \
args_point_robot_v1, args_hopper_param, args_walker_param
import numpy as np
import random

from models.encoder import RNNEncoder, MLPEncoder, SelfAttnEncoder
from algorithms.dqn import DQN
from algorithms.sac import SAC
from models.generative import CVAE
from environments.make_env import make_env
from torchkit import pytorch_utils as ptu
from torchkit.networks import FlattenMlp
from data_management.storage_policy import MultiTaskPolicyStorage
from utils import evaluation as utl_eval
from utils.tb_logger import TBLogger
from models.policy import TanhGaussianPolicy
from offline_learner import OfflineMetaLearner
from train_contrastive import FlatMLPEncoder, SelfAttentionEncoder
from utils.data_processing import sample_batch_data, sample_pos_neg_batch_data, preprocess_samples

import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
from sklearn import manifold

class OfflineLearner():
    
    def __init__(self, args, train_dataset, train_goal):
        self.dataset = train_dataset
        self.train_goal = train_goal
        
        self.env = make_env(args.env_name,
                            args.max_rollout_per_task,
                            seed = args.seed,
                            n_tasks = 1)
        self.initialize_policy()
        
        if self.args.log_tensorboard:
            self.tb_logger = TBLogger(self.args)
            
            
    def initialize_policy(self):
        if self.args.policy = 'iql':
            q1_network = FlattenMlp(input_size=self.args.obs_dim + self.args.action_dim,
                                    output_size=1,
                                    hidden_sizes=self.args.dqn_layers)
            q2_network = FlattenMlp(input_size=self.args.obs_dim + self.args.action_dim,
                                    output_size=1,
                                    hidden_sizes=self.args.dqn_layers)
            v_network = FlattenMlp(input_size = self.args.obs_dim,
                                   output_size = 1,
                                   hidden_sizes = self.args.dqn_layers)
            policy = TanhGaussianPolicy(obs_dim=self.args.obs_dim,
                                        action_dim=self.args.action_dim,
                                        hidden_sizes=self.args.policy_layers)
            self.agent = IQL(
                policy,
                q1_network,
                q2_network,
                v_network,

                actor_lr=self.args.actor_lr,
                critic_lr=self.args.critic_lr,
                gamma=self.args.gamma,
                tau=self.args.soft_target_tau,

                alpha_lr=self.args.alpha_lr,
                clip_grad_value=self.args.clip_grad_value,
                
                
            ).to(ptu.device)
        else:
            raise NotImplementedError
            
    def update(self):
        rl_losses_agg = {}
        for update in range(self.args.rl_update_per_iter):
            if self.args.log_train_time:
                _t_cost = time.time()
                
            obs, actions, rewards, next_obs, terms = self.sample_rl_batch(self.args.rl_batch_size)
            
            if self.args.policy == 'iql':
                rl_losses = self.agent.update(obs, actions, rewards, next_obs, terms)
            else:
                raise NotImplementedError
            

            for k, v in rl_losses.items():
                if update == 0:  # first iterate - create list
                    rl_losses_agg[k] = [v]
                else:  # append values
                    rl_losses_agg[k].append(v)
                    
        return rl_losses_agg
    
    def evaluate(self):
        
        num_episodes = self.args.max_rollouts_per_task
        num_steps_per_episode = self.env.unwrapped._max_episode_steps
        
        env = self.env
        obs = env.reset()
        obs = obs.reshape(-1, obs.shape[-1])
        step = 0
        
        returns_per_episode = np.zeros(num_episodes)
        
        for episode_idx in range(num_episodes):
            running_reward = 0.
            for step_idx in range(num_steps_per_episode):
                action, _, _, log_prob = self.agent.act(obs = obs, 
                                                        deterministic = self.args.eval_deterministic, 
                                                        return_log_prob = True)
                next_obs, reward, done, info = utl.env_step(eval_env, action.squeeze(dim=0))
                running_reward += reward.item()
                
                obs = next_obs.clone()
                steps+=1
            
            returns_per_episode[episode_idx] = running_reward
            
        return returns_per_episode
    
    def train(self):
        self._start_training()
        print('start training')
        for iter_ in range(self.args.num_iters):
            self.training_mode(True)
            train_stats = self.update()
            self.training_mode(False)
            self.log(iter_, train_stats)
    
    def training_mode(self, mode):
        self.agent.train(mode)
        
    def _start_training(self):
        self._n_rl_update_steps_total = 0
        self._start_time = time.time()
    
    def log(self, iteration, train_stats):
        
        
def main():
    parser - argparse.ArgumentParser()
    
    parser.add_argument('--env-type', default='gridworld_block')
    args, rest_args = parser.parse_known_args()
    env = args.env_type

    # --- GridWorld ---
    if env == 'gridworld_block':
        args = args_gridworld_block.get_args(rest_args)
    elif env == 'cheetah_vel':
        args = args_cheetah_vel.get_args(rest_args)
    elif env == 'point_robot':
        args = args_point_robot.get_args(rest_args)
    elif env == 'ant_dir':
        args = args_ant_dir.get_args(rest_args)
    elif env == 'point_robot_v1':
        args = args_point_robot_v1.get_args(rest_args)
    elif env == 'hopper_param':
        args = args_hopper_param.get_args(rest_args)
    elif env == 'walker_param':
        args = args_walker_param.get_args(rest_args)
    else:
        raise NotImplementedError


    set_gpu_mode(torch.cuda.is_available() and args.use_gpu)
    
    args, _ = off_utl.expand_args(args)
    
    unordered_dataset, goals = off_utl.load_dataset(data_dir=args.data_dir, args=args, arr_type='numpy')
    train_dataset = [unordered_data[0]]
    train_goal = [goals[0]]
    
    learner = OfflineLearner(args, train_dataset, train_goal)
    
    learner.train()
    
if __name__ = '__main__':
    main()