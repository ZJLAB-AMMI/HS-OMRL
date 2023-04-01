import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl
from torchkit import pytorch_utils as ptu
from torchkit.networks import Mlp, FlattenMlp

import numpy as np


# InfoNCE Loss
# q,k: (b, dim); neg: (b, N, dim)
def contrastive_loss(q, k, neg, infonce_temp = 1.0):
    N = neg.shape[1]
    b = q.shape[0]
    l_pos = torch.bmm(q.view(b, 1, -1), k.view(b, -1, 1)) 
    l_neg = torch.bmm(q.view(b, 1, -1), neg.transpose(1,2))
    logits = torch.cat([l_pos.view(b,1), l_neg.view(b,N)], dim = 1)
    
    labels = torch.zeros(b, dtype=torch.long)
    labels = labels.to(ptu.device)
    
    cross_entropy_loss = nn.CrossEntropyLoss()
    loss = cross_entropy_loss(logits/infonce_temp, lables)
    return loss

    
## Supervised Contrastive Learning Loss 
# ref: https://github.com/HobbitLong/SupContrast/blob/master/losses.py

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # print(mask.shape, anchor_count, contrast_count)
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
## hierarchical Multi Label Contrastive Loss
## ref: https://github.com/salesforce/hierarchicalContrastiveLearning/blob/master/losses/losses.py
    
class HMLCLoss(nn.Module):
    def __init__(self, temperature=0.07,
                 base_temperature=0.07, layer_penalty=None, loss_type='hmce'):
        super(HMLCLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        if not layer_penalty:
            self.layer_penalty = self.exp_L
        else:
            self.layer_penalty = layer_penalty
        self.sup_con_loss = SupConLoss(temperature)
        self.loss_type = loss_type

    def pow_2(self, value):
        return torch.pow( 2, torch.tensor(1/(value)).type(torch.float) )
    
    def exp_L(self, value, max_value):
        return torch.exp( torch.tensor(1/(max_value - value)).type(torch.float) )
    
    # def exp_L_inv(self, value, max_value):
    #     return torch.exp( torch.tensor )

    def forward(self, features, labels=None, mask=None):
        ## features: (batchsize, n_views, d)
        ## labels: (batchsize, n_labels)
        ## masks: (n_layer, batchsize, batchsize)
        ## return: scalar
        
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        masks = mask
        mask = torch.ones(labels.shape).to(device) if labels else None
        cumulative_loss = torch.tensor(0.0).to(device)
        max_loss_lower_layer = torch.tensor(float('-inf'))
        L = labels.shape[1] if labels else masks.shape[0]
        for l in range(0,L):
            if masks is None and labels is not None:
                mask[:, L-l:] = 0
                layer_labels = labels * mask
                mask_labels = torch.stack([torch.all(torch.eq(layer_labels[i], layer_labels), dim=1)
                                           for i in range(layer_labels.shape[0])]).type(torch.uint8).to(device)
            elif masks is not None and labels is None:
                mask_labels = masks[l]
            else:
                raise NotImplementedError
            layer_loss = self.sup_con_loss(features, mask=mask_labels)
            if self.loss_type == 'hmc':
                cumulative_loss += self.layer_penalty(l, L) * layer_loss
            elif self.loss_type == 'hce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                cumulative_loss += layer_loss
            elif self.loss_type == 'hmce':
                layer_loss = torch.max(max_loss_lower_layer.to(layer_loss.device), layer_loss)
                cumulative_loss += self.layer_penalty(l, L) * layer_loss
            else:
                raise NotImplementedError('Unknown loss')
            # _, unique_indices = unique(layer_labels, dim=0)
            max_loss_lower_layer = torch.max(
                max_loss_lower_layer.to(layer_loss.device), layer_loss)
            # labels = labels[unique_indices]
            # mask = mask[unique_indices]
            # features = features[unique_indices]
            
        return cumulative_loss / L
    
## Simple Siamese Loss
# ref: https://github.com/facebookresearch/simsiam
    
class SimSiamLoss(nn.Module):
    def __init__(self, loss_type = 'cossim', predictor = None):
        super(SimSiamLoss, self).__init__()
        self.loss_type = loss_type
        
        if predictor is None:
            self.predictor = nn.Identity()
        else:
            self.predictor = predictor 
            
        if loss_type == 'cossim':
            self.sim_func = nn.CosineSimilarity(dim=1)
        else:
            raise NotImplementedError
    
        
    def forward(self, features, predictor = None):
        ## features: (batchsize, n_views, d)
        
        b, n, d = features.shape
        anchor = features.reshape(-1, d) 
        
        if predictor is not None:
            x1 = predictor(anchor)
        else:
            x1 = self.predictor(anchor)
        x2 = anchor.detach().clone()
        x1 = x1.reshape(b,n,-1)
        x2 = x2.reshape(b,n,-1)
        
        p1 = x1[:,0,:].reshape(b,-1)
        p2 = x1[:,1,:].reshape(b,-1)
        z1 = x2[:,0,:].reshape(b,-1)
        z2 = x2[:,1,:].reshape(b,-1)        
        
        loss = - 0.5 * ( self.sim_func(p1, z2).mean() + self.sim_funcm(p2,z1).mean() )
        
        return loss
    
# class HardContrastiveLoss(nn.Module):
#     def __init__(self, alpha = 0.1, temperature = 1.0):
#         self.alpha = alpha
#         self.temperature = temperature
        
#     def forward(self, features):
#         raise NotImplementedError
        
    
class SimpleLoss(nn.Module):
    def __init__(self):
        super(SimpleLoss, self).__init__()
        
    def forward(self, features):
        ## features: (batchsize, n_views, d)
        raise NotImplementedError
        
class AlignmentLoss(nn.Module):
    def __init__(self):
        super(AlignmentLoss, self).__init__()
        
    def forward(self, features, alpha=2):
        ## features: (batchsize, n_views, d)
        b, n, d = features.shape
        
        x1 = features[:, 0, :]
        x2 = features[:, 1, :]
        # loss = torch.mean(torch.matmul(x1, x2.T))
        loss = (x1 - x2).norm(dim=1).pow(alpha).mean()
        
        return loss
    
class UniformityLoss(nn.Module):
    def __init__(self):
        super(UniformityLoss, self).__init__()
        
    def forward(self, features, t=2):
        ## features: (batchsize, n_views, d)
        b, n, d = features.shape
        
        x1 = features[:, 0, :]
        x2 = features[:, 1, :]
        # loss = torch.mean(torch.matmul(x1, x2.T))
        sq_pdist = torch.pdist(x1, p=2).pow(2)
        loss = torch.exp( -t*sq_pdist ).mean().log()
        
        return loss
    
class CombinedLoss(nn.Module):
    def __init__(self, temp = 0.2 ):
        super(CombinedLoss, self).__init__()
        self.alignment = AlignmentLoss()
        self.uniformity =  UniformityLoss()
        self.temp = temp
        
    def forward(self, features, **kwargs):
        loss = self.temp * self.uniformity(features) + (1-self.temp)*self.alignment(features)
        return loss
        
        
### Hard Contrastive Learning
#ref: https://github.com/joshr17/HCL/blob/main/image/main.py

# def get_negative_mask(batch_size):
    

class HardContrastiveLoss(nn.Module):
    def __init__(self, tau_plus=0.1, beta=1.0, temperature = 0.5, estimator = 'hard_negative'):
        # tau_plus: Positive class prior
        # beta: Choose Loss Function
        super(HardContrastiveLoss, self).__init__()
        self.tau_plus = tau_plus
        self.beta = beta
        self.temperature = temperature
        self.estimator = estimator
        
    def set_beta(self, beta):
        self.beta = beta
        
    def get_hard_negative(self, dist, neg_mask):
        # N, _ = neg.shape
        N = neg_mask.sum(-1)
        neg = neg_mask * dist
        imp = ( neg_mask * (self.beta * dist.log()).exp() ).detach()
        # imp = ( np.exp(self.beta) * neg ).detach()
        # imp = imp.detach()
        reweight_neg = (imp*neg).sum(-1) / imp.sum(-1) * neg_mask.sum(-1)
        # Ng = reweight_neg
        # Ng = (-self.tau_plus * N * pos + reweight_neg) / (1-self.tau_plus)
        # Constrain
        Ng = torch.clamp(reweight_neg, min = N*np.e**(-1/self.temperature))
        return Ng
    
    def get_hard_positive(self, dist, pos_mask, inverse_dist):
        N = pos_mask.sum(-1)
        pos = pos_mask * dist
        inverse_pos = pos_mask * inverse_dist
        imp = (pos_mask * (self.beta * inverse_dist.log()).exp() ).detach()
        # imp = (np.exp(self.beta) * inverse_pos).detach()
        # imp = imp.detach()
        reweight_pos = (imp*pos).sum(-1) / imp.sum(-1) * pos_mask.sum(-1)
        Ps = torch.clamp(reweight_pos, min = N*np.e**(-1/self.temperature))
        return Ps, imp
        
        
    def get_loss(self, numerator, denominator, pos_mask):
        loss = ( pos_mask * (-torch.log(numerator) + torch.log(denominator)) ).sum(dim=-1) / pos_mask.sum(dim=-1)
        return loss
    
    def forward(self, features, mask):
        
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        
        bsz, npos, dim = features.shape
        
        mask = mask.repeat(npos, npos)
        # Create logits mask
        # [[1-I(d), 1-I(d)],[1-I(d), 1-I(d)]]
        # logits_mask = 1 - torch.eye(bsz, dtype=torch.float32).to(device)
        # logits_mask = logits_mask.repeat(npos, npos)
        
        # Create self mask 
        # [1-I(2d)]
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(bsz * npos).view(-1, 1).to(device),
            0
        )
        
        # Create negative mask
        neg_mask = (1-mask) * logits_mask
        # Create Positive mask
        pos_mask = mask * logits_mask
        
        
        # Calculate Loss
        out = torch.cat(torch.unbind(features, dim=1), dim=0)
        log_dist = torch.mm(out, out.t().contiguous()) / self.temperature
        log_dist_max, _  = torch.max(log_dist, dim=1, keepdim=True)
        log_dist = log_dist - log_dist_max.detach()
        dist = torch.exp(log_dist)
        
        inverse_log_dist = - log_dist
        inverse_log_dist_max, _ = torch.max(inverse_log_dist, dim=1, keepdim=True)
        inverse_log_dist = inverse_log_dist - inverse_log_dist_max.detach()
        inverse_dist = torch.exp(inverse_log_dist)
        
        # dist = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        # log_dist = torch.log(dist)
        old_dist = dist.clone()
        neg = dist * neg_mask
        pos = dist * pos_mask
        
        inverse_pos = inverse_dist * pos_mask

        
        if self.estimator == 'hard_negative':
            N = bsz*2
            Ng = self.get_hard_negative(dist, neg_mask).reshape(-1,1)
            Ps = pos.sum(dim=-1, keepdim=True).reshape(-1,1)
            denominator = (Ps+Ng).expand(-1, N)
            numerator = dist
            
            loss = self.get_loss(numerator, denominator, pos_mask)
            # loss = ( pos_mask * (-torch.log(numerator) + torch.log(denominator)) ).sum(dim=-1) / pos_mask.sum(dim=-1)
            loss = loss.mean()
            
        elif self.estimator == 'hard_positive':
            N = bsz * 2
            Ng = self.get_hard_negative(dist, neg_mask).reshape(-1,1)
            Ps, pos_imp = self.get_hard_positive(dist, pos_mask, inverse_dist)
            Ps = Ps.reshape(-1,1)
            
            denominator = (Ps+Ng).expand(-1, N)
            numerator = dist
            # numerator = (pos * pos_imp) / pos_imp.sum(dim=-1)
            
            # loss = self.get_loss(numerator, denominator, pos_mask)
            loss = ( pos_mask * pos_imp * (-torch.log(numerator) + torch.log(denominator)) ).sum(dim=-1) / (pos_mask.sum(dim=-1) * pos_imp.sum(dim=-1))
            loss = loss.mean()
            
        elif self.estimator == 'easy':
            N = bsz * 2
            Ng = neg.sum(dim=-1, keepdim=True)
            Ps = pos.sum(dim=-1, keepdim=True)
            denominator = (Ps+Ng).expand(-1, N)
            numerator = dist
            
            loss = self.get_loss(numerator, denominator, pos_mask)
            loss = loss.mean()
                        
        else:
            raise NotImplementedError
        
        # loss = ( -torch.log( pos/(pos+Ng) ) ).mean()
        
        return loss
