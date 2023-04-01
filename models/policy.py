import numpy as np
import torch
from torch import nn as nn

from torchkit.policies_base import ExplorationPolicy
from torchkit.distributions import TanhNormal
from torchkit.networks import Mlp, FlattenMlp
from torchkit.core import np_ify, PyTorchModule



LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class TanhGaussianPolicy(Mlp, ExplorationPolicy):
    """
    Usage:
    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```
    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.
    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
            self,
            obs_dim,
            action_dim,
            hidden_sizes,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs, deterministic=False):
        actions = self.get_actions(obs, deterministic=deterministic)
        return actions[0, :], {}

    @torch.no_grad()
    def get_actions(self, obs, deterministic=False):
        outputs = self.forward(obs, deterministic=deterministic)[0]
        return np_ify(outputs)

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std
        # print(mean, log_std)
        log_prob = None
        if deterministic:
            action = torch.tanh(mean)
            if return_log_prob:
                tanh_normal = TanhNormal(mean, std)
                log_prob = tanh_normal.log_prob(action)
                log_prob = log_prob.sum(dim=1, keepdim=True)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return action, mean, log_std, log_prob

    
    
class FlattenMlpWithProjection(nn.Module):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 embedding_dim,
                 projection_dim,
                 output_size, 
                 hidden_size,
                ):
        super().__init__()
        self.context_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim),
        )
        self.flatten_policy = FlattenMlp(input_size=obs_dim+act_dim+projection_dim,
                                        output_size=output_size,
                                        hidden_sizes=hidden_size)
        
        
    def forward(self, *inputs, **kwargs):
        contexts = inputs[-1]
        projection = self.context_projection(contexts)
        new = (*inputs[:-1], projection)
        out = self.flatten_policy(*new, **kwargs)
        return out
        
        
class TanhGaussianPolicyWithProjection(nn.Module):
    def __init__(self, 
                obs_dim,
                act_dim,
                embedding_dim,
                projection_dim,
                hidden_size,
                **kwargs
                ):
        super().__init__()
        self.context_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, projection_dim),
        )
        self.policy = TanhGaussianPolicy(obs_dim = obs_dim+projection_dim,
                                        action_dim = act_dim,
                                        hidden_sizes = hidden_size)
        
    def forward(self, *inputs, **kwargs):
        obs, contexts = inputs
        projection = self.context_projection(contexts)
        inputs = torch.cat((obs, projection), dim=-1)
        action, mean, log_std, log_prob = self.policy(inputs, deterministic= True, return_log_prob=True)
        return action, mean, log_std, log_prob
        
'''    
def MLPwith(Mlp, ExplorationPolicy):
    def __init__(self,
                obs_dim,
                embedding_dim,
                action_dim,
                hidden_sizes,
                std=None,
                init_w=1e-3,
                **kwargs ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX
        
        
    def get_action(self, obs, deterministic=False):
        actions = self.get_actions(obs, deterministic=deterministic)
        return actions[0, :], {}

    @torch.no_grad()
    def get_actions(self, obs, deterministic=False):
        outputs = self.forward(obs, deterministic=deterministic)[0]
        return np_ify(outputs)
    
    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
'''