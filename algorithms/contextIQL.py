import os
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchkit.pytorch_utils as ptu
from torchkit.distributions import TanhNormal

def expectile_loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1-expectile))
    return weight * (diff **2)

class ContextIQL(nn.Module):
    def __init__(self,
                 policy,
                 q1_network,
                 q2_network,
                 v_network,

                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 gamma=0.99,
                 tau=5e-3,

                 use_cql=False,
                 alpha_cql=2.,
                 entropy_alpha=0.2,
                 automatic_entropy_tuning=True,
                 alpha_lr=3e-4,
                 clip_grad_value=None,
                 
                 expectile = 0.8,
                 normalize_advantage = False,
                 beta = 3.0,
                 max_weight = 100.,
                 normalize_actor_loss = False,
                 
                 encoder = None,
                 ):
        super().__init__()

        self.gamma = gamma
        self.tau = tau
        self.use_cql = use_cql    # Conservative Q-Learning loss
        self.alpha_cql = alpha_cql    # Conservative Q-Learning weight parameter
        self.automatic_entropy_tuning = automatic_entropy_tuning    # Wasn't tested
        self.clip_grad_value = clip_grad_value
        
        self.expectile = expectile
        self.normalize_advantage = normalize_advantage
        self.beta = beta
        self.max_weight = max_weight
        self.normalize_actor_loss = normalize_actor_loss
        
        self.encoder = encoder
        
        # q networks - use two network to mitigate positive bias
        self.qf1 = q1_network
        self.qf1_optim = Adam(self.qf1.parameters(), lr=critic_lr)

        self.qf2 = q2_network
        self.qf2_optim = Adam(self.qf2.parameters(), lr=critic_lr)        
        
        self.vf = v_network
        if self.encoder is not None:
            vf_params = list(self.vf.parameters()) + list(self.encoder.parameters())
        else:
            vf_params = self.vf.parameters()
        self.vf_optim = Adam(vf_params, lr=critic_lr)
        # self.vf_optim = Adam(self.vf.parameters(), lr=critic_lr)

        # target networks
        self.qf1_target = copy.deepcopy(self.qf1)
        self.qf2_target = copy.deepcopy(self.qf2)

        self.policy = policy
        self.policy_optim = Adam(self.policy.parameters(), lr=actor_lr)
        
        # self.policy_target = copy.deepcopy(self.policy)
        
        # # automatic entropy coefficient tuning
        # if self.automatic_entropy_tuning:
        #     # self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(ptu.device)).item()
        #     self.target_entropy = -self.policy.action_dim
        #     self.log_alpha_entropy = torch.zeros(1, requires_grad=True, device=ptu.device)
        #     self.alpha_entropy_optim = Adam([self.log_alpha_entropy], lr=alpha_lr)
        #     self.alpha_entropy = self.log_alpha_entropy.exp()
        # else:
        #     self.alpha_entropy = entropy_alpha

    def forward(self, obs, contexts=None):
        action, _, _, _ = self.policy(obs, contexts)
        q1, q2 = self.qf1(obs, action, contexts), self.qf2(obs, action, contexts)
        return action, q1, q2

    def act(self, obs, contexts, deterministic=False, return_log_prob=False):
        action, mean, log_std, log_prob = self.policy(obs, contexts, 
                                                      deterministic=deterministic,
                                                      return_log_prob=return_log_prob)
        return action, mean, log_std, log_prob

    def update(self, obs, action, reward, next_obs, done, contexts, **kwargs):
        # computation of critic loss
        #torch.autograd.set_detect_anomaly(True)
        # Compute Value Loss
        self.qf1.train()
        self.qf2.train()
        self.vf.train()
        self.policy.train()
        
        with torch.no_grad():
            self.qf1_target.eval()
            self.qf2_target.eval()
            target_q1 = self.qf1_target(obs, action, contexts)
            target_q2 = self.qf2_target(obs, action, contexts)
            target_q = torch.min(target_q1, target_q2)
            target_q = target_q.detach().clone()
        value = self.vf(obs, contexts)
        value_loss = expectile_loss( target_q - value, self.expectile ).mean()
        
        # Compute critic Loss
        with torch.no_grad():
            self.vf.eval()
            next_v = self.vf(next_obs, contexts)
            target_q = reward + self.gamma*next_v
            target_q = target_q.detach().clone()
        current_q1 = self.qf1(obs, action, contexts)
        current_q2 = self.qf2(obs, action, contexts)
        qf1_loss = ((current_q1 - target_q)**2).mean()
        qf2_loss = ((current_q2 - target_q)**2).mean()
        critic_loss = qf1_loss + qf2_loss
        
        # Compute actor loss
        advantage = torch.min(target_q1, target_q2) - value.detach()
        if self.normalize_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        weights = torch.exp(advantage * self.beta)
        weights = torch.clip(weights, max=self.max_weight)
        
        
        _, mean, log_std, _ = self.act(obs, contexts)#, return_log_prob = True)
        # print(mean, log_std)
        # normal_dist = torch.distributions.normal.Normal(mean, torch.exp(log_std))
        dist = TanhNormal(mean, torch.exp(log_std))
        # print(action)
        log_probs = dist.log_prob(action)
        # print(weights, log_probs)
        # print(torch.isinf(weights).any(), torch.isinf(log_probs).any())
        # print(weights.max(), weights.min())
        # print(log_probs.max(), log_probs.min())
        actor_loss = -(weights * log_probs)
        # print(actor_loss)
        # print(torch.isinf(actor_loss).any())
        # print(actor_loss.mean())
        if self.normalize_actor_loss:
            actor_loss = actor_loss / actor_loss.detach().abs().mean()
        actor_loss = actor_loss.mean()
        # print(value_loss, actor_loss, critic_loss)
        # Update Models
        self.vf_optim.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.vf.parameters(), 5.0)
        self.vf_optim.step()
        
        self.qf1_optim.zero_grad()
        self.qf2_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qf1.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm_(self.qf2.parameters(), 5.0)
        self.qf1_optim.step()
        self.qf2_optim.step()
        
        self.policy_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.policy_optim.step()
        
        self.soft_target_update()
        
        info = {'qf1_loss': qf1_loss.item(),
                'qf2_loss': qf2_loss.item(),
                'vf_loss': value_loss.item(),
                'policy_loss': actor_loss.item(),
        }
        
        return info
        

    def _min_q(self, obs, action, contexts):
        q1 = self.qf1(obs, action, contexts)
        q2 = self.qf2(obs, action, contexts)
        min_q = torch.min(q1, q2)
        return min_q

    def soft_target_update(self):
        ptu.soft_update_from_to(self.qf1, self.qf1_target, self.tau)
        ptu.soft_update_from_to(self.qf2, self.qf2_target, self.tau)
        # ptu.soft_update_from_to(self.policy, self.policy_target, self.tau)

    def _clip_grads(self, net):
        for p in net.parameters():
            p.grad.data.clamp_(-self.clip_grad_value, self.clip_grad_value)
    
#     def update_actor(self, obs, action, reward, next_obs, done, **kwargs):
#         # computation of actor loss
#         # bug fixed: this should be after the q function update
#         new_action, _, _, log_prob = self.act(obs, return_log_prob=True)
#         min_q_new_actions = self._min_q(obs, new_action)
#         policy_loss = ((self.alpha_entropy * log_prob) - min_q_new_actions).mean()


#         # update policy network
#         self.policy_optim.zero_grad()
#         policy_loss.backward()
#         if self.clip_grad_value is not None:
#             self._clip_grads(self.policy)
#         self.policy_optim.step()

#         if self.automatic_entropy_tuning:
#             alpha_entropy_loss = -(self.log_alpha_entropy * (log_prob + self.target_entropy).detach()).mean()

#             self.alpha_entropy_optim.zero_grad()
#             alpha_entropy_loss.backward()
#             self.alpha_optim.step()

#             self.alpha_entropy = self.log_alpha_entropy.exp()
#             # alpha_entropy_tlogs = self.alpha_entropy.clone()    # For TensorboardX logs
#         else:
#             alpha_entropy_loss = torch.tensor(0.).to(ptu.device)
#             # alpha_entropy_tlogs = torch.tensor(self.alpha_entropy)  # For TensorboardX logs

#         return {'policy_loss': policy_loss.item(), 'alpha_entropy_loss': alpha_entropy_loss.item()}