# train offpolicy rl with context-aggregator, after the pretraining of contrastive task encoder

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
args_point_robot_v1, args_hopper_param, args_walker_param, args_point_goal
import numpy as np
import random

from models.encoder import RNNEncoder, MLPEncoder, SelfAttnEncoder
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



# class 


class OfflineContrastive(OfflineMetaLearner):
    # algorithm class of offline meta-rl with relabelling

    def __init__(self, args, train_dataset, train_goals, eval_dataset, eval_goals, ood_eval_dataset, ood_eval_goals):
        """
        Seeds everything.
        Initialises: logger, environments, policy (+storage +optimiser).
        """

        self.args = args

        # make sure everything has the same seed
        utl.seed(self.args.seed)

        # initialize tensorboard logger
        if self.args.log_tensorboard:
            self.tb_logger = TBLogger(self.args)

        self.args, _ = off_utl.expand_args(self.args, include_act_space=True)
        if self.args.act_space.__class__.__name__ == "Discrete":
            self.args.policy = 'dqn'
        else:
            # self.args.policy = 'sac'
            self.args.policy = 'iql'

        # load augmented buffer to self.storage
        self.load_buffer(train_dataset, train_goals)
        if self.args.pearl_deterministic_encoder:
            self.args.augmented_obs_dim = self.args.obs_dim + self.args.task_embedding_size
        else:
            self.args.augmented_obs_dim = self.args.obs_dim + self.args.task_embedding_size * 2
        self.args.embedding_dim = self.args.task_embedding_size
        self.args.projection_dim = self.args.contrastive_embedding_size
        self.goals = train_goals
        self.eval_goals = eval_goals
        self.ood_eval_goals = ood_eval_goals
        # context set, to extract task encoding
        ## CHANGED WITH NEW PREPROCESSOR
        # preprocess the dataset
        # dataset = (data, trajectory_starts, policy_starts)
        # data: (n_tasks, n_samples, dim) (dim= state_dim*2 + action_dim + 1 + 1 + 1)
        # trajectory_starts, policy_starts: (n_tasks, num_trajectories/num_policies)
        
        self.context_dataset = preprocess_samples( train_dataset )
        self.eval_context_dataset = preprocess_samples( eval_dataset ) 
        self.ood_eval_context_dataset = preprocess_samples( ood_eval_dataset )


        # initialize policy
 
        # initialize task encoder
        self.encoder = SelfAttentionEncoder(
                hidden_size=self.args.aggregator_hidden_size,
                num_hidden_layers=2,
                task_embedding_size=self.args.task_embedding_size,
                projection_embedding_size = self.args.contrastive_embedding_size,
                action_size=self.args.act_space.n if self.args.act_space.__class__.__name__ == "Discrete" else self.args.action_dim,
                state_size=self.args.obs_dim,
                reward_size=1,
                term_size=0, # encode (s,a,r,s') only
                normalize=self.args.normalize_z,
        	).to(ptu.device)
        self.encoder.load_state_dict(torch.load(self.args.encoder_model_path, map_location=ptu.device))
        print(f'Encoder loaded from: {self.args.encoder_model_path}')
        
        self.initialize_policy(encoder = None)
        print('Policy Initialization Finished')
        # create environment for evaluation
        self.env = make_env(args.env_name,
                            args.max_rollouts_per_task,
                            seed=args.seed,
                            n_tasks=self.args.num_eval_tasks)
        # fix the possible eval goals to be the testing set's goals
        self.env.set_all_goals(eval_goals)

        # create env for eval on training tasks
        self.env_train = make_env(args.env_name,
                            args.max_rollouts_per_task,
                            seed=args.seed,
                            n_tasks=self.args.num_train_tasks)
        self.env_train.set_all_goals(train_goals)
        self.env_ood = make_env(args.env_name,
                            args.max_rollouts_per_task,
                            seed=args.seed,
                            n_tasks=self.args.num_ood_eval_tasks)
        self.env_ood.set_all_goals(ood_eval_goals)

        #if self.args.env_name == 'GridNavi-v2' or self.args.env_name == 'GridBlock-v2':
        #    self.env.unwrapped.goals = [tuple(goal.astype(int)) for goal in self.goals]

        '''
        if self.args.relabel_type == 'gt':
            # create an env for reward/transition relabelling
            self.relabel_env = make_env(args.env_name,
                            args.max_rollouts_per_task,
                            seed=args.seed,
                            n_tasks=1)
        elif self.args.relabel_type == 'generative':
            self.generative_model = CVAE(
            	hidden_size=args.cvae_hidden_size,
                num_hidden_layers=args.cvae_num_hidden_layers,
            	z_dim=self.args.task_embedding_size,
                action_size=self.args.act_space.n if self.args.act_space.__class__.__name__ == "Discrete" else self.args.action_dim,
                state_size=self.args.obs_dim,
                reward_size=1).to(ptu.device)
            self.generative_model.load_state_dict(torch.load(self.args.generative_model_path, 
                map_location=ptu.device))
            self.generative_model.train(False)
            print('generative model loaded from {}'.format(self.args.generative_model_path))
        else: 
            raise NotImplementedError
        '''

        #self._preprocess_positive_samples()

        #print(self.evaluate())
        #self.vis_sample_embeddings('test.png')
        #sys.exit(0)
#     def load_buffer(self, train_dataset, train_goals):
#         # process obs, actions, ... into shape (num_trajs*num_timesteps, dim) for each task
#         dataset = []
#         for i, set in enumerate(train_dataset):
#             obs, actions, rewards, next_obs, terminals, traj_start, policy_start = set
            
#             device=ptu.device
#             obs = ptu.FloatTensor(obs).to(device)
#             actions = ptu.FloatTensor(actions).to(device)
#             rewards = ptu.FloatTensor(rewards).to(device)
#             next_obs = ptu.FloatTensor(next_obs).to(device)
#             terminals = ptu.FloatTensor(terminals).to(device)
#             traj_start = ptu.FloatTensor(traj_start).to(device)
#             policy_start = ptu.FloatTensor(policy_start).to(device)

#             obs = obs.transpose(0, 1).reshape(-1, obs.shape[-1])
#             actions = actions.transpose(0, 1).reshape(-1, actions.shape[-1])
#             rewards = rewards.transpose(0, 1).reshape(-1, rewards.shape[-1])
#             next_obs = next_obs.transpose(0, 1).reshape(-1, next_obs.shape[-1])
#             terminals = terminals.transpose(0, 1).reshape(-1, terminals.shape[-1])

#             obs = ptu.get_numpy(obs)
#             actions = ptu.get_numpy(actions)
#             rewards = ptu.get_numpy(rewards)
#             next_obs = ptu.get_numpy(next_obs)
#             terminals = ptu.get_numpy(terminals)
#             traj_start = ptu.get_numpy(traj_start)
#             policy_start = ptu.get_numpy(policy_start)

#             dataset.append([obs, actions, rewards, next_obs, terminals, traj_start, policy_start])

#         #augmented_obs_dim = dataset[0][0].shape[1]

#         self.storage = MultiTaskPolicyStorage(max_replay_buffer_size=dataset[0][0].shape[0],
#                                               obs_dim=dataset[0][0].shape[1],
#                                               action_space=self.args.act_space,
#                                               tasks=range(len(train_goals)),
#                                               trajectory_len=self.args.trajectory_len)

#         for task, set in enumerate(dataset):
#             self.storage.add_samples(task,
#                                      observations=set[0],
#                                      actions=set[1],
#                                      rewards=set[2],
#                                      next_observations=set[3],
#                                      terminals=set[4],
#                                      new_trajectories=set[5],
#                                      new_policies=set[6],
#                                     )
#         return #train_goals, augmented_obs_dim
        
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
        
    def sample_context_batch(self, batch_size, tasks = None, trainset = 'train', context_len=10, percentile = [0,1]):
        # Sample a batch of data of shape (n_tasks, batch_size, context_len, dim)
        if trainset == 'train':
            dataset = self.context_dataset
        elif trainset == 'eval':
            dataset = self.eval_context_dataset
        elif trainset == 'ood':
            dataset = self.ood_eval_context_dataset
        else:
             raise NotImplementedError
                
        sampled_data, tasks = sample_batch_data(dataset, batch_size, context_len=context_len, tasks=tasks, percentile=percentile)
        return sampled_data, tasks
        
    def update(self, tasks, iter_num=0):
        rl_losses_agg = {}
        if self.args.log_train_time:
	        time_cost = {'data_sampling':0, 'negatives_sampling':0, 'update_encoder':0, 'update_rl':0}

        for update in range(self.args.rl_updates_per_iter):
            if self.args.log_train_time:
                _t_cost = time.time()
            
            # sample rl batch, context batch and update agent
            # sample random RL batch
            obs, actions, rewards, next_obs, terms = self.sample_rl_batch(tasks, self.args.rl_batch_size) # [task, batch, dim]
            # sample corresponding context batch
            ### NEW CHANGED!!!!
            sampled_data, tasks = self.sample_context_batch(batch_size=1, tasks=tasks, context_len = 100) # [ts'=ts*num_context_traj, task, dim]

            
            if not len(sampled_data.shape)>3:
                n_tasks, n_samples, n_dim = sampled_data.shape
                # sampled_data = sampled_data[rank, :, :]
                # tasks = tasks[rank]
                sampled_data = sampled_data.reshape(n_tasks*n_samples, n_dim)
            else:
                n_tasks, n_samples, n_context, n_dim = sampled_data.shape
                # sampled_data = sampled_data[rank,:,:,:]
                # tasks = tasks[rank]
                sampled_data = sampled_data.reshape(n_tasks*n_samples, n_context, n_dim)
            
            with torch.no_grad():
                _, encodings = self.encoder(sampled_data)
            
            # encoding = self.context_encoder(encodings)
            # task_encoding = encoding.unsqueeze(1)
            # self.context_encoder_optimizer.zero_grad()
            t, d = encodings.size()
            encodings = encodings.unsqueeze(1)
            encodings = encodings.expand(t, self.args.rl_batch_size, d) # [task, batch(repeat), dim]
            t, b, _ = encodings.size()
            contexts = encodings.reshape(t * b, -1)
            # obs = torch.cat((obs, encodings), dim=-1)
            # next_obs = torch.cat((next_obs, encodings), dim=-1) # [task, batch, obs_dim+z_dim]

            # flatten out task dimension
            t, b, _ = obs.size()
            obs = obs.view(t * b, -1)
            actions = actions.view(t * b, -1)
            rewards = rewards.view(t * b, -1)
            next_obs = next_obs.view(t * b, -1)
            terms = terms.view(t * b, -1)
            #print('forward: q learning')
            # RL update (Q learning)
            #rl_losses = self.agent.update(obs, actions, rewards, next_obs, terms, action_space=self.env.action_space)
            if self.args.policy == 'dqn':
                rl_losses = self.agent.update(obs, actions, rewards, next_obs, terms)
                # if not self.args.use_additional_task_info:
                    # self.context_encoder_optimizer.step()
            elif self.args.policy == 'sac':
                rl_losses = self.agent.update_critic(obs, actions, rewards, next_obs, terms, action_space=self.env.action_space)
                # if not self.args.use_additional_task_info:
                    # self.context_encoder_optimizer.step()
                obs = obs.detach()
                next_obs = next_obs.detach()
                actor_losses = self.agent.update_actor(obs, actions, rewards, next_obs, terms, action_space=self.env.action_space)
                rl_losses.update(actor_losses)
            elif self.args.policy == 'iql':
                obs = obs.detach()
                next_obs = next_obs.detach()
                # print('update iteration: ', update)
                # print(contexts.shape, obs.shape)
                rl_losses = self.agent.update(obs, actions, rewards, next_obs, terms, contexts)
            else:
                raise NotImplementedError

            '''
            if self.args.log_train_time:
                _t_now = time.time()
                time_cost['update_rl'] += (_t_now-_t_cost)
                _t_cost = _t_now
            '''
            if self.args.use_additional_task_info:
                rl_losses['task_pred_loss'] = task_pred_loss.item()


            for k, v in rl_losses.items():
                if update == 0:  # first iterate - create list
                    rl_losses_agg[k] = [v]
                else:  # append values
                    rl_losses_agg[k].append(v)
        # take mean
        for k in rl_losses_agg:
            rl_losses_agg[k] = np.mean(rl_losses_agg[k])
        self._n_rl_update_steps_total += self.args.rl_updates_per_iter

        if self.args.log_train_time:
            print(time_cost)

        return rl_losses_agg

    
    def evaluate(self, trainset='train', percentile = None):
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
        if self.args.policy == 'sac' or self.args.policy == 'iql':
            log_probs = np.zeros((num_tasks, self.args.trajectory_len))

        # eval_env = self.env_train if trainset else self.env
        for task in eval_env.unwrapped.get_all_task_idx():
            obs = ptu.from_numpy(eval_env.reset(task))
            obs = obs.reshape(-1, obs.shape[-1])
            step = 0

            for episode_idx in range(num_episodes):
                running_reward = 0.
                
                sampled_data, task = self.sample_context_batch(batch_size=1, tasks=[task], trainset=trainset, context_len=100, percentile = percentile)
            
                if not len(sampled_data.shape)>3:
                    n_tasks, n_samples, n_dim = sampled_data.shape
                    sampled_data = sampled_data.reshape(n_tasks*n_samples, n_dim)
                else:
                    n_tasks, n_samples, n_context, n_dim = sampled_data.shape
                    sampled_data = sampled_data.reshape(n_tasks*n_samples, n_context, n_dim)

                with torch.no_grad():
                    _, contexts = self.encoder(sampled_data)

                observations[task, step, :] = ptu.get_numpy(obs[0, :obs_size])
                
                for step_idx in range(num_steps_per_episode):
                    # add distribution parameters to observation - policy is conditioned on posterior
                    # augmented_obs = torch.cat((obs, task_desc), dim=-1)
                    if self.args.policy == 'dqn':
                        action, value = self.agent.act(obs=augmented_obs, deterministic=True)
                    else:
                        action, _, _, log_prob = self.agent.act(obs=obs, contexts = contexts,
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
        if self.args.save_model and (iteration % self.args.save_interval == 0):
            save_path = os.path.join(self.tb_logger.full_output_folder, 'models')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            torch.save(self.agent.state_dict(), os.path.join(save_path, "agent{0}.pt".format(iteration)))
            torch.save(self.encoder.state_dict(), os.path.join(save_path, "encoder{0}.pt".format(iteration)))
            if hasattr(self, 'context_encoder'):
                torch.save(self.context_encoder.state_dict(), os.path.join(save_path, 
                    "context_encoder{0}.pt".format(iteration)))

        if iteration % self.args.log_interval == 0:
            if self.args.policy == 'dqn':
                returns, success_rate, observations, rewards, reward_preds = self.evaluate()
                returns_train, success_rate_train, observations_train, rewards_train, reward_preds_train = self.evaluate(trainset='train')
            # This part is super specific for the Semi-Circle env
            # elif self.args.env_name == 'PointRobotSparse-v0':
            #     returns, success_rate, log_probs, observations, \
            #     rewards, reward_preds, reward_belief, reward_belief_discretized, points = self.evaluate()
            else:
                returns, success_rate, log_probs, observations, rewards, reward_preds = self.evaluate(trainset='eval')
                returns_train, success_rate_train, log_probs_train, observations_train, rewards_train, reward_preds_train = self.evaluate(trainset='train')
                returns_ood, success_rate_ood, log_probs_ood, observations_ood, rewards_ood, reward_preds_ood = self.evaluate(trainset='ood')

            if self.args.log_tensorboard:
                if self.args.env_name == 'GridBlock-v2':
                    tasks_to_vis = np.random.choice(self.args.num_eval_tasks, 5)
                    for i, task in enumerate(tasks_to_vis):
                        self.env.reset(task)
                        self.tb_logger.writer.add_figure('policy_vis_test/task_{}'.format(i),
                                                         utl_eval.plot_rollouts(observations[task, :], self.env),
                                                         self._n_rl_update_steps_total)
                    tasks_to_vis = np.random.choice(self.args.num_train_tasks, 5)
                    for i, task in enumerate(tasks_to_vis):
                        self.env_train.reset(task)
                        self.tb_logger.writer.add_figure('policy_vis_train/task_{}'.format(i),
                                                        utl_eval.plot_rollouts(observations_train[task, :], self.env_train),
                                                        self._n_rl_update_steps_total)


                if self.args.max_rollouts_per_task > 1:
                    '''
                    for episode_idx in range(self.args.max_rollouts_per_task):
                        self.tb_logger.writer.add_scalar('returns_multi_episode/episode_{}'.
                                                         format(episode_idx + 1),
                                                         np.mean(returns[:, episode_idx]),
                                                         self._n_rl_update_steps_total)
                    self.tb_logger.writer.add_scalar('returns_multi_episode/sum',
                                                     np.mean(np.sum(returns, axis=-1)),
                                                     self._n_rl_update_steps_total)
                    self.tb_logger.writer.add_scalar('returns_multi_episode/success_rate',
                                                     np.mean(success_rate),
                                                     self._n_rl_update_steps_total)
                    '''
                    raise NotImplementedError
                else:
                    self.tb_logger.writer.add_scalar('returns/returns_mean', np.mean(returns),
                                                     self._n_rl_update_steps_total)
                    self.tb_logger.writer.add_scalar('returns/returns_std', np.std(returns),
                                                     self._n_rl_update_steps_total)
                    self.tb_logger.writer.add_scalar('returns/success_rate', np.mean(success_rate),
                                                     self._n_rl_update_steps_total)

                    self.tb_logger.writer.add_scalar('returns_train/returns_mean', np.mean(returns_train),
                                                     self._n_rl_update_steps_total)
                    self.tb_logger.writer.add_scalar('returns_train/returns_std', np.std(returns_train),
                                                     self._n_rl_update_steps_total)
                    self.tb_logger.writer.add_scalar('returns_train/success_rate', np.mean(success_rate_train),
                                                     self._n_rl_update_steps_total)
                    
                    self.tb_logger.writer.add_scalar('returns_ood/returns_mean', np.mean(returns_ood),
                                                     self._n_rl_update_steps_total)
                    self.tb_logger.writer.add_scalar('returns_ood/returns_std', np.std(returns_ood),
                                                     self._n_rl_update_steps_total)
                    self.tb_logger.writer.add_scalar('returns_ood/success_rate', np.mean(success_rate_ood),
                                                     self._n_rl_update_steps_total)
                    
                    self.tb_logger.writer.add_scalar('generalization/generalisation_error', np.mean(returns_train) - np.mean(returns),
                                                     self._n_rl_update_steps_total)
                    self.tb_logger.writer.add_scalar('generalization/generalisation_ood_error', np.mean(returns_train) - np.mean(returns_ood), 
                                                     self._n_rl_update_steps_total)
                    

                if self.args.policy == 'dqn':
                    self.tb_logger.writer.add_scalar('rl_losses/qf_loss_vs_n_updates', train_stats['qf_loss'],
                                                     self._n_rl_update_steps_total)
                    self.tb_logger.writer.add_scalar('weights/q_network',
                                                     list(self.agent.qf.parameters())[0].mean(),
                                                     self._n_rl_update_steps_total)
                    if list(self.agent.qf.parameters())[0].grad is not None:
                        param_list = list(self.agent.qf.parameters())
                        self.tb_logger.writer.add_scalar('gradients/q_network',
                                                         sum([param_list[i].grad.mean() for i in
                                                              range(len(param_list))]),
                                                         self._n_rl_update_steps_total)
                    self.tb_logger.writer.add_scalar('weights/q_target',
                                                     list(self.agent.target_qf.parameters())[0].mean(),
                                                     self._n_rl_update_steps_total)
                    if list(self.agent.target_qf.parameters())[0].grad is not None:
                        param_list = list(self.agent.target_qf.parameters())
                        self.tb_logger.writer.add_scalar('gradients/q_target',
                                                         sum([param_list[i].grad.mean() for i in
                                                              range(len(param_list))]),
                                                         self._n_rl_update_steps_total)
                    # other loss terms
                    for k in train_stats.keys():
                        if k != 'qf_loss':
                            self.tb_logger.writer.add_scalar('rl_losses/'+k, train_stats[k], 
                                self._n_rl_update_steps_total)
                else:
                    self.tb_logger.writer.add_scalar('policy/log_prob', np.mean(log_probs),
                                                     self._n_rl_update_steps_total)
                    self.tb_logger.writer.add_scalar('rl_losses/qf1_loss', train_stats['qf1_loss'],
                                                     self._n_rl_update_steps_total)
                    self.tb_logger.writer.add_scalar('rl_losses/qf2_loss', train_stats['qf2_loss'],
                                                     self._n_rl_update_steps_total)
                    self.tb_logger.writer.add_scalar('rl_losses/policy_loss', train_stats['policy_loss'],
                                                     self._n_rl_update_steps_total)
                    # self.tb_logger.writer.add_scalar('rl_losses/alpha_entropy_loss', train_stats['alpha_entropy_loss'],
                    #                                  self._n_rl_update_steps_total)

                    # other loss terms
                    for k in train_stats.keys():
                        if k not in ['qf1_loss', 'qf2_loss', 'policy_loss', 'alpha_entropy_loss']:
                            self.tb_logger.writer.add_scalar('rl_losses/'+k, train_stats[k], 
                                self._n_rl_update_steps_total)

                    # weights and gradients
                    self.tb_logger.writer.add_scalar('weights/q1_network',
                                                     list(self.agent.qf1.parameters())[0].mean(),
                                                     self._n_rl_update_steps_total)
                    if list(self.agent.qf1.parameters())[0].grad is not None:
                        param_list = list(self.agent.qf1.parameters())
                        self.tb_logger.writer.add_scalar('gradients/q1_network',
                                                         sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                                                         self._n_rl_update_steps_total)
                    self.tb_logger.writer.add_scalar('weights/q1_target',
                                                     list(self.agent.qf1_target.parameters())[0].mean(),
                                                     self._n_rl_update_steps_total)
                    if list(self.agent.qf1_target.parameters())[0].grad is not None:
                        param_list = list(self.agent.qf1_target.parameters())
                        self.tb_logger.writer.add_scalar('gradients/q1_target',
                                                         sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                                                         self._n_rl_update_steps_total)
                    self.tb_logger.writer.add_scalar('weights/q2_network',
                                                     list(self.agent.qf2.parameters())[0].mean(),
                                                     self._n_rl_update_steps_total)
                    if list(self.agent.qf2.parameters())[0].grad is not None:
                        param_list = list(self.agent.qf2.parameters())
                        self.tb_logger.writer.add_scalar('gradients/q2_network',
                                                         sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                                                         self._n_rl_update_steps_total)
                    self.tb_logger.writer.add_scalar('weights/q2_target',
                                                     list(self.agent.qf2_target.parameters())[0].mean(),
                                                     self._n_rl_update_steps_total)
                    if list(self.agent.qf2_target.parameters())[0].grad is not None:
                        param_list = list(self.agent.qf2_target.parameters())
                        self.tb_logger.writer.add_scalar('gradients/q2_target',
                                                         sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                                                         self._n_rl_update_steps_total)
                    self.tb_logger.writer.add_scalar('weights/policy',
                                                     list(self.agent.policy.parameters())[0].mean(),
                                                     self._n_rl_update_steps_total)
                    if list(self.agent.policy.parameters())[0].grad is not None:
                        param_list = list(self.agent.policy.parameters())
                        self.tb_logger.writer.add_scalar('gradients/policy',
                                                         sum([param_list[i].grad.mean() for i in range(len(param_list))]),
                                                         self._n_rl_update_steps_total)

            print("Iteration -- {}, Avg. return -- {:.3f}, \
                Avg. return train -- {:.3f}, Avg. return ood -- {:.3f}, Elapsed time {:5d}[s]"
                .format(iteration, np.mean(np.sum(returns, axis=-1)),
                    np.mean(np.sum(returns_train, axis=-1)), np.mean(np.sum(returns_ood, axis=-1)), 
                    int(time.time() - self._start_time)), train_stats)

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env-type', default='gridworld')
    # parser.add_argument('--env-type', default='point_robot_sparse')
    # parser.add_argument('--env-type', default='cheetah_vel')
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
    elif env == 'point_goal':
        args = args_point_goal.get_args(rest_args)
    else:
        raise NotImplementedError


    set_gpu_mode(torch.cuda.is_available() and args.use_gpu)

    args, _ = off_utl.expand_args(args) # add env information to args
    #print(args)

    unordered_dataset, goals = off_utl.load_dataset(data_dir=args.data_dir, args=args, arr_type='numpy')
    
    if env == 'cheetah_vel' or env == 'ant_dir':
        indexs = np.argsort(np.squeeze(goals))
        dataset = [unordered_dataset[i] for i in indexs]
        goals = goals[indexs]
    elif env == 'point_robot_v1' or 'point_goal':
        indexs = np.argsort(np.squeeze(goals[:,1]))
        dataset = [unordered_dataset[i] for i in indexs]
        goals = goals[indexs]
    else:
        raise NotImplementedError
    
    # assert args.num_train_tasks + args.num_eval_tasks + args.num_ood_eval_tasks == len(goals)
    if args.num_eval_tasks != 0 and args.num_ood_eval_tasks != 0:
        np.random.seed(args.numpy_seed)
        iid = args.num_train_tasks+args.num_eval_tasks
        iid_dataset, iid_goals = dataset[0:iid], goals[0:iid]
        ood_eval_dataset, ood_eval_goals = dataset[iid:], goals[iid:]
        permuted_iid = np.random.permutation(iid)
        train_id = permuted_iid[0:args.num_train_tasks]
        eval_id = permuted_iid[args.num_train_tasks:]
        train_dataset = [iid_dataset[i] for i in train_id]
        train_goals = iid_goals[train_id]
        eval_dataset = [iid_dataset[i] for i in eval_id]
        eval_goals = iid_goals[eval_id]
    else:
        np.random.seed(args.numpy_seed)
        iid = args.num_train_tasks
        iid_dataset, iid_goals = dataset[0:iid], goals[0:iid]
        train_dataset = eval_dataset = ood_eval_dataset = iid_dataset
        train_goals = eval_goals = ood_eval_goals = iid_goals
        args.num_eval_tasks = args.num_train_tasks
        args.num_ood_eval_tasks = args.num_train_tasks
    
    # train_dataset, train_goals = dataset[0:args.num_train_tasks], goals[0:args.num_train_tasks]
    # eval_dataset, eval_goals = dataset[args.num_train_tasks:], goals[args.num_train_tasks:]
    
    # dataset, goals = off_utl.load_dataset(data_dir=args.data_dir, args=args, arr_type='numpy')
    # assert args.num_train_tasks + args.num_eval_tasks == len(goals)
    # train_dataset, train_goals = dataset[0:args.num_train_tasks], goals[0:args.num_train_tasks]
    # eval_dataset, eval_goals = dataset[args.num_train_tasks:], goals[args.num_train_tasks:]
    
    print('Data Loaded')
    learner = OfflineContrastive(args, train_dataset, train_goals, eval_dataset, eval_goals, ood_eval_dataset, ood_eval_goals)
    learner.train()


if __name__ == '__main__':
    main()
