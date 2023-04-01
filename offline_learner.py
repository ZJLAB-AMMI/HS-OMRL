# code base for offline meta learning

import os
import sys
import time
import argparse
import torch
from torchkit.pytorch_utils import set_gpu_mode
#from models.vae import VAE
#from offline_metalearner import OfflineMetaLearner
import utils.config_utils as config_utl
from utils import helpers as utl, offline_utils as off_utl
from offline_rl_config import args_gridworld_block
import numpy as np

from models.encoder import RNNEncoder, MLPEncoder
from algorithms.dqn import DQN
from algorithms.sac import SAC
from algorithms.iql import IQL
from algorithms.contextIQL import ContextIQL
from environments.make_env import make_env
from torchkit import pytorch_utils as ptu
from torchkit.networks import FlattenMlp
from data_management.storage_policy import MultiTaskPolicyStorage
from utils import evaluation as utl_eval
from utils.tb_logger import TBLogger
from models.policy import TanhGaussianPolicy, TanhGaussianPolicyWithProjection, FlattenMlpWithProjection


class OfflineMetaLearner:
    '''
    general offline meta learner class
    the encoder takes a context batch [n_episodes*n_ts, dim] and outputs a task encoding z
    the agent makes decision conditioned on s and z
    __init__() and update() are not implemented by default, for different algorithms
    '''

    def __init__(self, **kwargs):
        raise NotImplementedError


    def initialize_policy(self, encoder = None):
        if self.args.policy == 'dqn':
            q_network = FlattenMlp(input_size=self.args.augmented_obs_dim,
                                   output_size=self.args.act_space.n,
                                   hidden_sizes=self.args.dqn_layers)
            self.agent = DQN(
                q_network,
                # optimiser_vae=self.optimizer_vae,
                lr=self.args.policy_lr,
                gamma=self.args.gamma,
                tau=self.args.soft_target_tau,
            ).to(ptu.device)
        elif self.args.policy == 'sac':
            # assert self.args.act_space.__class__.__name__ == "Box", (
            #     "Can't train SAC with discrete action space!")
            q1_network = FlattenMlp(input_size=self.args.augmented_obs_dim + self.args.action_dim,
                                    output_size=1,
                                    hidden_sizes=self.args.dqn_layers)
            q2_network = FlattenMlp(input_size=self.args.augmented_obs_dim + self.args.action_dim,
                                    output_size=1,
                                    hidden_sizes=self.args.dqn_layers)
            policy = TanhGaussianPolicy(obs_dim=self.args.augmented_obs_dim,
                                        action_dim=self.args.action_dim,
                                        hidden_sizes=self.args.policy_layers)
            self.agent = SAC(
                policy,
                q1_network,
                q2_network,

                actor_lr=self.args.actor_lr,
                critic_lr=self.args.critic_lr,
                gamma=self.args.gamma,
                tau=self.args.soft_target_tau,

                use_cql=self.args.use_cql if 'use_cql' in self.args else False,
                alpha_cql=self.args.alpha_cql if 'alpha_cql' in self.args else None,
                entropy_alpha=self.args.entropy_alpha,
                automatic_entropy_tuning=self.args.automatic_entropy_tuning,
                alpha_lr=self.args.alpha_lr,
                clip_grad_value=self.args.clip_grad_value,
            ).to(ptu.device)
        elif self.args.policy == 'iql':
            q1_network = FlattenMlpWithProjection(obs_dim = self.args.obs_dim,
                                                 act_dim = self.args.action_dim,
                                                 embedding_dim = self.args.embedding_dim,
                                                 projection_dim = self.args.projection_dim,
                                                 output_size = 1,
                                                 hidden_size = self.args.dqn_layers)
            q2_network = FlattenMlpWithProjection(obs_dim = self.args.obs_dim,
                                                 act_dim = self.args.action_dim,
                                                 embedding_dim = self.args.embedding_dim,
                                                 projection_dim = self.args.projection_dim,
                                                 output_size = 1,
                                                 hidden_size = self.args.dqn_layers)
            v_network = FlattenMlpWithProjection(obs_dim = self.args.obs_dim,
                                                 act_dim = 0,
                                                 embedding_dim = self.args.embedding_dim,
                                                 projection_dim = self.args.projection_dim,
                                                 output_size = 1,
                                                 hidden_size = self.args.dqn_layers)
            policy = TanhGaussianPolicyWithProjection(obs_dim = self.args.obs_dim,
                                                 act_dim = self.args.action_dim,
                                                 embedding_dim = self.args.embedding_dim,
                                                 projection_dim = self.args.projection_dim,
                                                 hidden_size = self.args.dqn_layers)
            # q1_network = FlattenMlp(input_size=self.args.augmented_obs_dim + self.args.action_dim,
            #                         output_size=1,
            #                         hidden_sizes=self.args.dqn_layers)
            # q2_network = FlattenMlp(input_size=self.args.augmented_obs_dim + self.args.action_dim,
            #                         output_size=1,
            #                         hidden_sizes=self.args.dqn_layers)
            # v_network = FlattenMlp(input_size = self.args.augmented_obs_dim,
            #                        output_size = 1,
            #                        hidden_sizes = self.args.dqn_layers)
            # policy = TanhGaussianPolicy(obs_dim=self.args.augmented_obs_dim,
            #                             action_dim=self.args.action_dim,
            #                             hidden_sizes=self.args.policy_layers)
            self.agent = ContextIQL(
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
                
                encoder = encoder
            ).to(ptu.device)
        else: 
            raise NotImplementedError

    # convert the training set to the multitask replay buffer
    def load_buffer(self, train_dataset, train_goals):
        # process obs, actions, ... into shape (num_trajs*num_timesteps, dim) for each task
        dataset = []
        for i, set in enumerate(train_dataset):
            obs, actions, rewards, next_obs, terminals,  traj_start, policy_start = set
            
            device=ptu.device
            obs = ptu.FloatTensor(obs).to(device)
            actions = ptu.FloatTensor(actions).to(device)
            rewards = ptu.FloatTensor(rewards).to(device)
            next_obs = ptu.FloatTensor(next_obs).to(device)
            terminals = ptu.FloatTensor(terminals).to(device)
            traj_start = ptu.FloatTensor(traj_start).to(device)
            policy_start = ptu.FloatTensor(policy_start).to(device)

            if len(obs.shape)>2:
                obs = obs.transpose(0, 1).reshape(-1, obs.shape[-1])
                actions = actions.transpose(0, 1).reshape(-1, actions.shape[-1])
                rewards = rewards.transpose(0, 1).reshape(-1, rewards.shape[-1])
                next_obs = next_obs.transpose(0, 1).reshape(-1, next_obs.shape[-1])
                terminals = terminals.transpose(0, 1).reshape(-1, terminals.shape[-1])
            # traj_start = traj_start.transpose(0, 1).reshape()

            obs = ptu.get_numpy(obs)
            actions = ptu.get_numpy(actions)
            rewards = ptu.get_numpy(rewards)
            next_obs = ptu.get_numpy(next_obs)
            terminals = ptu.get_numpy(terminals)
            traj_start = ptu.get_numpy(traj_start)
            policy_start = ptu.get_numpy(policy_start)


            dataset.append([obs, actions, rewards, next_obs, terminals, traj_start, policy_start])

        #augmented_obs_dim = dataset[0][0].shape[1]

        max_replay_buffer_size = len(np.nonzero(dataset[0][5])[0]) * self.args.trajectory_len
        self.storage = MultiTaskPolicyStorage(max_replay_buffer_size=max_replay_buffer_size,
                                              obs_dim=dataset[0][0].shape[1],
                                              action_space=self.args.act_space,
                                              tasks=range(len(train_goals)),
                                              trajectory_len=self.args.trajectory_len)

        for task, set in enumerate(dataset):
            self.storage.add_samples(task,
                                     observations=set[0],
                                     actions=set[1],
                                     rewards=set[2],
                                     next_observations=set[3],
                                     terminals=set[4],
                                     new_trajectories=set[5],
                                     new_policies=set[6],)
        return #train_goals, augmented_obs_dim


    # training offline RL, with evaluation on fixed eval tasks
    def train(self):
        self._start_training()
        print('start training')
        for iter_ in range(self.args.num_iters):
            self.training_mode(True)
            indices = np.random.choice(len(self.goals), self.args.meta_batch) # sample with replacement! it is important for FOCAL
            #print('training')
            train_stats = self.update(indices, iter_num=iter_)

            self.training_mode(False)
            #print('logging')
            self.log(iter_, train_stats)

    def update(self, tasks):
        raise NotImplementedError

    # do policy evaluation on eval tasks
    # trainset: evaluate on training tasks or testing tasks?
    def evaluate(self, trainset='train'):
        num_episodes = self.args.max_rollouts_per_task
        num_steps_per_episode = self.env.unwrapped._max_episode_steps
        if trainset == 'train':
            num_tasks = self.args.num_train_tasks
            eval_env = self.env_train
        elif trainset == 'eval':
            num_tasks = self.args.num_eval_tasks
            eval_env = self.env
        elif trainset == 'ood':
            num_tasks = self.args.num_ood_eval_tasks
            eval_env = self.env_ood
        # num_tasks = self.args.num_train_tasks if trainset else self.args.num_eval_tasks
        obs_size = self.env.unwrapped.observation_space.shape[0]

        returns_per_episode = np.zeros((num_tasks, num_episodes))
        success_rate = np.zeros(num_tasks)

        rewards = np.zeros((num_tasks, self.args.trajectory_len))
        reward_preds = np.zeros((num_tasks, self.args.trajectory_len))
        observations = np.zeros((num_tasks, self.args.trajectory_len + 1, obs_size))
        if self.args.policy == 'sac':
            log_probs = np.zeros((num_tasks, self.args.trajectory_len))

        # eval_env = self.env_train if trainset else self.env
        for task in eval_env.unwrapped.get_all_task_idx():
            obs = ptu.from_numpy(eval_env.reset(task))
            obs = obs.reshape(-1, obs.shape[-1])
            step = 0

            obs_context, actions_context, rewards_context, next_obs_context, terms_context = self.sample_context_batch([task], trainset=trainset)

            #task_desc = self.encoder.context_encoding(obs=obs_context, actions=actions_context, 
            #	rewards=rewards_context, next_obs=next_obs_context, terms=terms_context)
            n_timesteps, batch_size, _ = obs_context.shape
            task_desc = self.encoder(
                    obs=obs_context.reshape(n_timesteps*batch_size, -1), 
                    action=actions_context.reshape(n_timesteps*batch_size, -1), 
                    reward=rewards_context.reshape(n_timesteps*batch_size, -1), 
                    next_obs=next_obs_context.reshape(n_timesteps*batch_size, -1),
                ).view(n_timesteps, batch_size, -1).transpose(0,1)
            #print(task_desc.shape)
            task_desc = self.context_encoder(task_desc)
            #print(task_desc.shape)

            observations[task, step, :] = ptu.get_numpy(obs[0, :obs_size])

            for episode_idx in range(num_episodes):
                running_reward = 0.
                for step_idx in range(num_steps_per_episode):
                    # add distribution parameters to observation - policy is conditioned on posterior
                    augmented_obs = torch.cat((obs, task_desc), dim=-1)
                    if self.args.policy == 'dqn':
                        action, value = self.agent.act(obs=augmented_obs, deterministic=True)
                    else:
                        action, _, _, log_prob = self.agent.act(obs=augmented_obs,
                                                                deterministic=self.args.eval_deterministic,
                                                                return_log_prob=True)

                    # observe reward and next obs
                    next_obs, reward, done, info = utl.env_step(eval_env, action.squeeze(dim=0))
                    running_reward += reward.item()
                    # done_rollout = False if ptu.get_numpy(done[0][0]) == 0. else True
                    # update encoding
                    #task_sample, task_mean, task_logvar, hidden_state = self.update_encoding(obs=next_obs,
                    #                                                                         action=action,
                    #                                                                         reward=reward,
                    #                                                                         done=done,
                    #                                                                         hidden_state=hidden_state)
                    rewards[task, step] = reward.item()
                    #reward_preds[task, step] = ptu.get_numpy(
                    #    self.vae.reward_decoder(task_sample, next_obs, obs, action)[0, 0])

                    observations[task, step + 1, :] = ptu.get_numpy(next_obs[0, :obs_size])
                    if self.args.policy != 'dqn':
                        log_probs[task, step] = ptu.get_numpy(log_prob[0])

                    if "is_goal_state" in dir(eval_env.unwrapped) and eval_env.unwrapped.is_goal_state():
                        success_rate[task] = 1.
                    # set: obs <- next_obs
                    obs = next_obs.clone()
                    step += 1

                returns_per_episode[task, episode_idx] = running_reward

        # reward_preds is 0 here
        if self.args.policy == 'dqn':
            return returns_per_episode, success_rate, observations, rewards, reward_preds
        else:
            return returns_per_episode, success_rate, log_probs, observations, rewards, reward_preds

    def log(self, iteration, train_stats):
        # --- save model ---
        raise NotImplementedError

    def sample_rl_batch(self, tasks, batch_size):
        ''' sample batch of unordered rl training data from a list/array of tasks '''
        # this batch consists of transitions sampled randomly from replay buffer
        batches = [ptu.np_to_pytorch_batch(
            self.storage.random_batch(task, batch_size)) for task in tasks]
        unpacked = [utl.unpack_batch(batch) for batch in batches]
        # group elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    # sample num_context_trajs trajectories in buffer for each task, as task context
    # trainset: if true, tasks are in context_dataset, else, tasks are in eval_context_dataset
    # 4/22 updated: random sample i_episodes for each sampled task (to make sure that even tasks are sampled with replacement,
    # the sampled context can be different) 
    def sample_context_batch(self, tasks, trainset=True):
        raise NotImplementedError
#         if trainset:
#             contextset = self.context_dataset
#         else:
#             contextset = self.eval_context_dataset

#         #i_episodes = np.random.choice(contextset[0][0].shape[1], self.args.num_context_trajs)
#         context = []
#         for i in tasks:
#             i_episodes = np.random.choice(contextset[0][0].shape[1], self.args.num_context_trajs) # should be randomized at every task
#             context_i = [ptu.FloatTensor(contextset[i][j][:, i_episodes, :]).transpose(0,1).reshape(
#                 -1, contextset[i][j].shape[-1]) for j in range(len(contextset[i]))] # obs, act, reward, next_obs, term
#             context.append(context_i)

#         ret = [torch.stack([context[i][j] for i in range(len(tasks))], dim=0).transpose(0,1) for j in range(len(contextset[i]))]
#         return ret

    # random sample positive samples (query, key) with size (batchsize, [s,a,r,s',t])
    # return [s,a,r,s',t], [s,a,r,s',t]
    def sample_positive_pairs(self, batch_size, trainset=True):
        raise NotImplementedError
#         if trainset:
#             contextset = self.context_dataset
#         else:
#             contextset = self.eval_context_dataset

#         queries, keys = [[] for i in range(5)], [[] for i in range(5)]
#         #tasks = []
#         for i in range(batch_size):
#             i_task = np.random.randint(0, len(contextset))
#             #tasks.append(i_task)
#             i_q = np.random.randint(0, contextset[0][0].shape[0])
#             i_k = np.random.randint(0, contextset[0][0].shape[0])
#             j_q = np.random.randint(0, contextset[0][0].shape[1])
#             j_k = np.random.randint(0, contextset[0][0].shape[1])
#             for j in range(5):
#                 queries[j].append(ptu.FloatTensor(contextset[i_task][j][i_q, j_q]))
#                 keys[j].append(ptu.FloatTensor(contextset[i_task][j][i_k, j_k]))
        
#         queries = [torch.stack(i) for i in queries]
#         keys = [torch.stack(i) for i in keys]
#         return queries, keys#, tasks



    def _start_training(self):
        self._n_rl_update_steps_total = 0
        self._start_time = time.time()

    def training_mode(self, mode):
        self.agent.train(mode)
        self.encoder.train(mode)
        if hasattr(self, 'context_encoder'):
            self.context_encoder.train(mode)

    def load_models(self, **kwargs):
        if "agent_path" in kwargs and "encoder_path" in kwargs:
            self.agent.load_state_dict(torch.load(kwargs["agent_path"], map_location=ptu.device))
            self.encoder.load_state_dict(torch.load(kwargs["encoder_path"], map_location=ptu.device))
            if "context_encoder_path" in kwargs:
                self.context_encoder.load_state_dict(torch.load(kwargs["context_encoder_path"], map_location=ptu.device))
        elif "num_iter" in kwargs and "load_dir" in kwargs:
            save_path = kwargs["load_dir"]
            agent_path = os.path.join(save_path, "agent{0}.pt".format(kwargs["num_iter"]))
            encoder_path = os.path.join(save_path, "encoder{0}.pt".format(kwargs["num_iter"]))
            self.agent.load_state_dict(torch.load(agent_path, map_location=ptu.device))
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=ptu.device))
            if hasattr(self, 'context_encoder'):
                context_encoder_path = os.path.join(save_path, "context_encoder{0}.pt".format(kwargs["num_iter"]))
                self.context_encoder.load_state_dict(torch.load(context_encoder_path, map_location=ptu.device))
        else:
            raise NotImplementedError

        self.training_mode(False)

