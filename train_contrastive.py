# ablation: train contrastive transition representation without generative model

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchkit.pytorch_utils import set_gpu_mode
import utils.config_utils as config_utl
from utils import helpers as utl, offline_utils as off_utl
from offline_rl_config import args_gridworld_block, args_cheetah_vel, args_ant_dir, args_hopper_param, args_walker_param, args_point_robot_v1, args_point_goal
import numpy as np
import random

from models.encoder import RNNEncoder, MLPEncoder, SelfAttnEncoder
from algorithms.dqn import DQN
from algorithms.sac import SAC
from algorithms.iql import IQL
from models.generative import CVAE
from environments.make_env import make_env
from torchkit import pytorch_utils as ptu
from torchkit.networks import FlattenMlp
from data_management.storage_policy import MultiTaskPolicyStorage
from utils import evaluation as utl_eval
from utils.tb_logger import TBLogger
from models.policy import TanhGaussianPolicy
from offline_learner import OfflineMetaLearner
from losses.losses import SupConLoss, HMLCLoss, SimSiamLoss, AlignmentLoss, UniformityLoss, CombinedLoss, HardContrastiveLoss
from utils.data_processing import sample_batch_data, sample_pos_neg_batch_data, preprocess_samples

import matplotlib.pyplot as plt
#import matplotlib.colors as mcolors
from sklearn import manifold


class FlatMLPEncoder(MLPEncoder):
    def __init__(self,
                 # network size
                 hidden_size=64,
                 num_hidden_layers=2,
                 task_embedding_size=32,
                 # actions, states, rewards
                 action_size=2,
                 state_size=2,
                 reward_size=1,
                 term_size=1,
                 normalize=False
                 ):
        super(FlatMLPEncoder, self).__init__(hidden_size, 
                                            num_hidden_layers, 
                                            task_embedding_size,
                                            action_size,
                                            state_size,
                                            reward_size,
                                            term_size,
                                            normalize,
                                            Flatten=False)
        
    # input state transition sample, output task embedding
    def forward(self, inputs):
        assert inputs.shape[-1] == self.state_size*2 + self.action_size + self.reward_size + self.term_size
    
        out = self.encoder(inputs)
        if not self.normalize:
            return out
        else:
            return F.normalize(out)
        
        
class SelfAttentionEncoder(FlatMLPEncoder):
    def __init__(self,
                 # network size
                 hidden_size=64,
                 num_hidden_layers=2,
                 task_embedding_size=32,
                 projection_embedding_size=8,
                 # actions, states, rewards
                 action_size=2,
                 state_size=2,
                 reward_size=1,
                 term_size=1,
                 normalize=True,
                 # aggregator hyperparameter
                 context_length = 10, 
                 ):
        super(SelfAttentionEncoder, self).__init__(hidden_size, 
                                            num_hidden_layers, 
                                            task_embedding_size,
                                            action_size,
                                            state_size,
                                            reward_size,
                                            term_size,
                                            normalize,
                                           )

        self.attention = nn.MultiheadAttention(task_embedding_size, num_heads = 1, batch_first=True)
        # self.projection_head = nn.Linear(task_embedding_size, projected_embedding_size)
        self.projection_head = nn.Sequential(
                                    nn.Linear(task_embedding_size, task_embedding_size),
                                    nn.ReLU(),
                                    nn.Linear(task_embedding_size, projection_embedding_size),
                                )
    # input (b, N, dim), output (b, dim)
    def forward(self, inp):
        bsz, c_len, dim = inp.shape
        inp = inp.reshape(-1, dim)
        encoded_inp = self.embed_forward(inp)
        _, encoded_dim = encoded_inp.shape
        encoded_inp = encoded_inp.reshape(bsz, c_len, encoded_dim)
        out = self.attention_forward(encoded_inp)
        projected_out = self.projection_forward(out)
        
        return projected_out, out
    
    def embed_forward(self, inputs):
        assert inputs.shape[-1] == self.state_size*2 + self.action_size + self.reward_size + self.term_size
    
        out = self.encoder(inputs)
        if not self.normalize:
            return out
        else:
            return F.normalize(out)
    
    def attention_forward(self, inputs):
        # input (b, L, dim), output (b, embed_dim)
        attended_output, _ = self.attention(inputs,inputs,inputs)
        out = torch.mean(attended_output, dim=1, keepdim=False)
        # out = self.final_mlp(attended_output)
        if not self.normalize:
            return out
        else:
            return F.normalize(out)
    
    def projection_forward(self, inputs):
        projected_out = self.projection_head(inputs)
        if not self.normalize:
            return projected_out
        else:
            return F.normalize(projected_out)        
    

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
            self.args.policy = 'sac'

        # load augmented buffer to self.storage
        self.load_buffer(train_dataset, train_goals)
        if self.args.pearl_deterministic_encoder:
            self.args.augmented_obs_dim = self.args.obs_dim + self.args.task_embedding_size
        else:
            self.args.augmented_obs_dim = self.args.obs_dim + self.args.task_embedding_size * 2
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
        self.ood_eval_context_datset = preprocess_samples( ood_eval_dataset)
        # initialize policy
        self.initialize_policy()
        
        ###### NEW ADDED!!!!!
        # initialize loss
        # self.contrastive_loss = SupConLoss(temperature = self.args.infonce_temp)
        if self.args.contrastive_loss == 'combine':
            self.contrastive_loss = CombinedLoss( temp = self.args.infonce_temp )
        elif self.args.contrastive_loss == 'hard' or self.args.contrastive_loss == 'hard_neg':
            self.contrastive_loss = HardContrastiveLoss( temperature = self.args.infonce_temp, estimator = 'hard_negative', beta = self.args.beta)
        elif self.args.contrastive_loss == 'hard_pos':
            self.contrastive_loss = HardContrastiveLoss( temperature = self.args.infonce_temp, estimator = 'hard_positive', beta = self.args.beta)
        elif self.args.contrastive_loss == 'easy':
            self.contrastive_loss = HardContrastiveLoss( temperature = self.args.infonce_temp, estimator = 'easy', beta = self.args.beta)
        elif self.args.n_label_layer == 1:
            self.contrastive_loss = SupConLoss(temperature = self.args.infonce_temp)
        elif self.args.n_label_layer > 1:
            self.contrastive_loss = HMLCLoss(temperature = self.args.infonce_temp)
        elif self.args.n_label_layer == 0:
            # self.predictor_network = 
            self.contrastive_loss = SimSiamLoss()
        # initialize task encoder
        self.uniform = UniformityLoss()
        self.alignment = AlignmentLoss()
        
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
        
        


        #else:
        #    raise NotImplementedError
        if not self.args.n_label_layer == 0:
            self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.args.encoder_lr)
        # else:
        #     self.predictor = 

        # context encoder: convert (batch, N, dim) to (batch, dim)
        # self.context_encoder = SelfAttnEncoder(input_dim=self.args.task_embedding_size,
        #     num_output_mlp=self.args.context_encoder_output_layers).to(ptu.device)
        # self.context_encoder_optimizer = torch.optim.Adam(self.context_encoder.parameters(), lr=self.args.encoder_lr)


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

        # self._preprocess_positive_samples()

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
        
    ## NEW ADDED !!!!!!
    
    def sample_contrastive_batch(self, batch_size, trainset = True, split_type = 'SupCL', n_layer=1):
        # Sample a batch of contrastive data of shape (batch_size, 2, context_len, dim)
        sampled_data, task_label_masks = sample_pos_neg_batch_data( self.context_dataset, batch_size , context_len=100, n_layer=n_layer, split_type = split_type) #self.args.context_len)
        # if sampled_data.shape[2] == 1:
        #     sampled_data = sampled_data.squeeze(axis=2)
        return sampled_data, task_label_masks

    def sample_context_batch(self, batch_size, tasks = None, trainset = 'train', context_len = 10):
        # Sample a batch of data of shape (n_tasks, batch_size, context_len, dim)
        if trainset == 'train':
            dataset = self.context_dataset
        elif trainset == 'eval':
            dataset = self.eval_context_dataset
        elif trainset == 'ood':
            dataset = self.ood_eval_context_datset
        else:
             raise NotImplementedError
                
        sampled_data, tasks = sample_batch_data(dataset, batch_size, context_len=context_len, tasks = None)
        # if sampled_data.shape[2] == 1:
        #     sampled_data = sampled_data.squeeze(axis=2)
        return sampled_data, tasks
        

    def update(self, tasks, iter_num=0):
        rl_losses_agg = {}
        if self.args.log_train_time:
	        time_cost = {'data_sampling':0, 'negatives_sampling':0, 'update_encoder':0, 'update_rl':0}
            
        if self.args.beta_annealing:
            beta_now = iter_num / self.args.num_iters * self.beta
            self.contrastive_loss.set_beta(beta_now)

        for update in range(self.args.rl_updates_per_iter):
            if self.args.log_train_time:
                _t_cost = time.time()
            #print('data sampling')
            
            # sample key, query, negative samples and train encoder
            # sampled_data: (batchsize, 2 (2 is n_views), context_len, dim) 
            # task_label_mask: (batchsize, batchsize)
            # mask[i,j] = 1 if sample_i and sample_j from the same task, 0 otherwise
            # always, mask[i,i] = 1
            batch_size = self.args.contrastive_batch_size
            sampled_data, task_label_mask = self.sample_contrastive_batch(batch_size, split_type=self.args.layer_type, n_layer=self.args.n_label_layer)
            eval_sampled_data, eval_task_label_mask = self.sample_contrastive_batch(batch_size, split_type= 'SupCL' , n_layer = 1)

            if self.args.log_train_time:
                _t_now = time.time()
                time_cost['data_sampling'] += (_t_now-_t_cost)
                _t_cost = _t_now

            n_views = sampled_data.shape[1]
            if not len(sampled_data.shape)>3:
                data_dim = sampled_data.shape[2]
                sampled_data = sampled_data.reshape(batch_size * n_views, data_dim)
                encoded_data, task_embedding = self.encoder.forward( sampled_data )
                encoded_dim = encoded_data.shape[-1]
                encoded_data = encoded_data.reshape(batch_size, n_views, encoded_dim)
                embedding_dim = task_embedding.shape[-1]
                task_embedding = task_embedding.reshape(batch_size, n_views, embedding_dim)
                
                eval_data_dim = eval_sampled_data.shape[2]
                eval_sampled_data = eval_sampled_data.reshape(batch_size * n_views, eval_data_dim)
                eval_encoded_data, eval_task_embedding  = self.encoder.forward( eval_sampled_data )
                eval_encoded_dim = eval_encoded_data.shape[-1]
                eval_encoded_data = eval_encoded_data.reshape(batch_size, n_views, eval_encoded_dim)
                eval_task_embedding = eval_task_embedding.reshape(batch_size, n_views, -1)
                
            else:
                context_len = sampled_data.shape[2]
                data_dim = sampled_data.shape[3]
                sampled_data = sampled_data.reshape(batch_size *n_views, context_len, data_dim)
                encoded_data, task_embedding = self.encoder.forward( sampled_data )
                encoded_dim = encoded_data.shape[-1]
                encoded_data = encoded_data.reshape(batch_size, n_views, encoded_dim)
                task_embedding = task_embedding.reshape(batch_size, n_views, -1)
                
                eval_context_len = eval_sampled_data.shape[2]
                eval_data_dim = eval_sampled_data.shape[3]
                eval_sampled_data = eval_sampled_data.reshape(batch_size *n_views, eval_context_len, eval_data_dim)
                eval_encoded_data, eval_task_embedding = self.encoder.forward( eval_sampled_data )
                eval_encoded_dim = eval_encoded_data.shape[-1]
                eval_encoded_data = eval_encoded_data.reshape(batch_size, n_views, eval_encoded_dim)
                eval_task_embedding = eval_task_embedding.reshape(batch_size, n_views, -1)
            
            if self.args.contrastive_loss == 'combine':
                contrastive_loss = self.contrastive_loss( encoded_data, mask = task_label_mask )
            else:
                contrastive_loss = self.contrastive_loss( encoded_data, mask = task_label_mask )
            
            # NEW ADDED
            # Regularization to reduce the dependency of z on input x
            # regular_loss = None
            # all_loss = contrastive_loss + regular_loss
            
            
            self.encoder_optimizer.zero_grad()
            contrastive_loss.backward()
            self.encoder_optimizer.step()
            
            eval_contrastive_loss = self.contrastive_loss( eval_encoded_data, mask = eval_task_label_mask )
            eval_uniformity_loss_projected = self.uniform( eval_encoded_data )
            eval_alignment_loss_projected = self.alignment( eval_encoded_data )
            eval_uniformity_loss_embedded = self.uniform( eval_task_embedding )
            eval_alignment_loss_embedded = self.alignment( eval_task_embedding )

            if self.args.log_train_time:
                _t_now = time.time()
                time_cost['update_encoder'] += (_t_now-_t_cost)
                _t_cost = _t_now

            
            
            rl_losses = {'contrastive_loss':contrastive_loss.item(),
                        'eval_contastive': eval_contrastive_loss.item(),
                        'eval_uniform_after_projection': eval_uniformity_loss_projected.item(),
                        'eval_alignment_after_projection': eval_alignment_loss_projected.item(),
                        'eval_uniform_before_projection': eval_uniformity_loss_embedded.item(),
                        'eval_alignment_before_projection': eval_alignment_loss_embedded.item(),
                        }
            
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


    def log(self, iteration, train_stats):
        #super().log(iteration, train_stats)
        if self.args.save_model and (iteration % self.args.save_interval == 0):
            save_path = os.path.join(self.tb_logger.full_output_folder, 'models')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            torch.save(self.encoder.state_dict(), os.path.join(save_path, "encoder{0}.pt".format(iteration)))

        if iteration % self.args.log_interval == 0:
            if self.args.log_tensorboard:
                for k in train_stats.keys():
                    self.tb_logger.writer.add_scalar('rl_losses/'+k, train_stats[k], 
                        self._n_rl_update_steps_total)
            print("Iteration -- {}, Elapsed time {:5d}[s]"
                .format(iteration, int(time.time() - self._start_time)), train_stats)

        # visualize embeddings
        # if self.args.log_tensorboard and (iteration % self.args.log_vis_interval == 0):
        #     save_path = os.path.join(self.tb_logger.full_output_folder, 'vis_z')
        #     if not os.path.exists(save_path):
        #         os.mkdir(save_path)
        #     try:
        #         self.vis_sample_embeddings(os.path.join(save_path, "train_fig{0}.png".format(iteration)), trainset='train')
        #         self.vis_sample_embeddings(os.path.join(save_path, "test_fig{0}.png".format(iteration)), trainset='eval')
        #         self.vis_sample_embeddings(os.path.join(save_path, "ood_fig{0}.png".format(iteration)), trainset='ood')
        #     except:
        #         pass


    # visualize the encodings of (s,a,r,s')
    # distinguish different tasks' critical samples and unimportant samples with different colors
    # use tsne
    def vis_sample_embeddings(self, save_path, trainset='train'):
        self.training_mode(False)
        if trainset == 'train':
            goals = self.goals
        elif trainset == 'eval':
            goals = self.eval_goals
        elif trainset == 'ood':
            goals = self.ood_eval_goals
        # goals = self.goals if trainset else self.eval_goals
        goals = np.squeeze(goals)
        # rank = np.argsort(goals)
        rank = np.arange(len(goals))
        x, y = [], []
        
        sampled_data, tasks = self.sample_context_batch( self.args.contrastive_batch_size , trainset=trainset, context_len=100)
        if not len(sampled_data.shape)>3:
            n_tasks, n_samples, n_dim = sampled_data.shape
            sampled_data = sampled_data[rank, :, :]
            tasks = tasks[rank]
            sampled_data = sampled_data.reshape(n_tasks*n_samples, n_dim)
        else:
            n_tasks, n_samples, n_context, n_dim = sampled_data.shape
            sampled_data = sampled_data[rank,:,:,:]
            tasks = tasks[rank]
            sampled_data = sampled_data.reshape(n_tasks*n_samples, n_context, n_dim)
        _, encodings = self.encoder(sampled_data)
        encodings = ptu.get_numpy(encodings)
        tasks = np.repeat( tasks, n_samples)
        x, y = encodings, tasks
            
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity = int(np.sqrt(n_tasks * n_samples)) )
        X_tsne = tsne.fit_transform(np.asarray(x))

        x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
        data = (X_tsne - x_min) / (x_max - x_min)

        if self.args.env_name == 'GridBlock-v2':
            colors = plt.cm.rainbow(np.linspace(0,1,len(goals)+1))
        else:
            colors = plt.cm.rainbow(np.linspace(0,1,len(goals)))
        #print(colors)

        plt.cla()
        fig = plt.figure()
        ax = plt.subplot(111)
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], str(y[i]),
                    color=colors[y[i]], #plt.cm.Set1(y[i] / 21),
                    fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.savefig(save_path)



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
    elif env == 'ant_dir':
        args = args_ant_dir.get_args(rest_args)
    elif env == 'hopper_param':
        args = args_hopper_param.get_args(rest_args)
    elif env == 'walker_param':
        args = args_walker_param.get_args(rest_args)
    elif env == 'point_robot_v1':
        args = args_point_robot_v1.get_args(rest_args)
    elif env == 'point_goal':
        args = args_point_goal.get_args(rest_args)
    else:
        raise NotImplementedError


    set_gpu_mode(torch.cuda.is_available() and args.use_gpu)

    args, _ = off_utl.expand_args(args) # add env information to args
    #print(args)
    
    unordered_dataset, goals = off_utl.load_dataset(data_dir=args.data_dir, args=args, arr_type='numpy')
    
    if env == 'cheetah_vel' or env == 'ant_dir' or env == 'walker_param' or env == 'hopper_param':
        indexs = np.argsort(np.squeeze(goals))
        dataset = [unordered_dataset[i] for i in indexs]
        goals = goals[indexs]
    elif env == 'point_robot_v1' or 'point_goal':
        indexs = np.argsort(np.squeeze(goals[:,1]))
        dataset = [unordered_dataset[i] for i in indexs]
        goals = goals[indexs]
    else:
        raise NotImplementedError
    # print()
    assert args.num_train_tasks + args.num_eval_tasks + args.num_ood_eval_tasks == len(goals)
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

    # dataset, goals = off_utl.load_dataset(data_dir=args.data_dir, args=args, arr_type='numpy')
    # print(args.num_train_tasks, args.num_eval_tasks, len(goals))
    # print(goals.shape)
    # assert args.num_train_tasks + args.num_eval_tasks == len(goals)
    # train_dataset, train_goals = dataset[0:args.num_train_tasks], goals[0:args.num_train_tasks]
    # eval_dataset, eval_goals = dataset[args.num_train_tasks:], goals[args.num_train_tasks:]

    learner = OfflineContrastive(args, train_dataset, train_goals, eval_dataset, eval_goals, ood_eval_dataset, ood_eval_goals)
    learner.train()


if __name__ == '__main__':
    main()
