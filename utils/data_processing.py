import os
import numpy as np
import gym
import torch
from utils import helpers as utl
import matplotlib.pyplot as plt
from torchkit import pytorch_utils as ptu
from utils import offline_utils as outl
from scipy.linalg import block_diag

## input: dataset: {n_tasks, (obs, act, rew, next_obs, term, trajectory_starts, policy_starts)}
def batch2traj(dataset):
    ## output: traj_dataset: {n_tasks, n_traj}
    traj_dataset = []
    for data in dataset:
        size = obs.shape[0]
        obs, act, rew, next_obs, term, trajectory_starts, _ = data
        traj_split = np.where(trajectory_starts)
        n_obs, n_act, n_rew, n_next_obs, n_term = [],[],[],[],[]
        length = []
        for ts, te in zip(traj_split[:-1], traj_split[1:]):
            n_obs.append( obs[ts:te] )
            n_act.append( act[ts:te] )
            n_rew.append( rew[ts:te] )
            n_next_obs.append( next_obs[ts:te] )
            n_term.append( term[ts:te] )
            length.append( te-ts )
            
        last = traj_split[-1]
        n_obs.append( obs[last:] )
        n_act.append( act[last:] )
        n_rew.append( rew[last:] )
        n_next_obs.append( next_obs[last:] )
        n_term.append( term[last:] )
        length.append( size-last )
    
        traj_dataset.append( [n_obs, n_act, n_rew, n_next_obs, n_term, length] )
        
    return traj_dataset

def batch2policy(dataset):
    policy_dataset = []
    for data in dataset:
        size = obs.shape[0]
        obs, act, rew, next_obs, term, _, policy_starts = data
        traj_split = np.where(policy_starts)
        n_obs, n_act, n_rew, n_next_obs, n_term = [],[],[],[],[]
        length = []
        for ts, te in zip(traj_split[:-1], traj_split[1:]):
            n_obs.append( obs[ts:te] )
            n_act.append( act[ts:te] )
            n_rew.append( rew[ts:te] )
            n_next_obs.append( next_obs[ts:te] )
            n_term.append( term[ts:te] )
            length.append( te-ts )
            
        last = traj_split[-1]
        n_obs.append( obs[last:] )
        n_act.append( act[last:] )
        n_rew.append( rew[last:] )
        n_next_obs.append( next_obs[last:] )
        n_term.append( term[last:] )
        length.append( size-last )
    
        policy_dataset.append( [n_obs, n_act, n_rew, n_next_obs, n_term, length] )
        
    return policy_dataset



# process the training dataset for fast positives sampling
def preprocess_samples(dataset):
    # input: n_tasks * (obs, act, rew, next_obs, term, trajectory_starts, policy_starts)
    # each element: np.array([traj_len, n_traj, dim])
    # Outputs:
    #     ds: [n_tasks, n_sample, dim]
    #     ts: [n_tasks, List: traj_split_points]
    #     ps: [n_tasks, List: policy_split_points]
    ds, ts, ps = [],[],[]
    for task_data in dataset:
        # print([d.shape for d in task_data])
        task_ds = np.concatenate(task_data[0:4], axis=-1)
        shape = task_ds.shape
        task_ds = task_ds.reshape( -1, shape[-1] )
        ## task_ds shape: n_sample_per_task * sum(dim)
        
        traj_start = np.nonzero(task_data[5])[0]
        policy_start = np.nonzero(task_data[6])[0]
        ds.append(task_ds)
        ts.append(traj_start)
        ps.append(policy_start)
        
    # ds = np.array(ds)
    return (ds, ts, ps)
    
    
def sample_batch_data(dataset, batchsize, tasks=None, context_len = 1, arr_type = 'torch', percentile = [0,1]):
    # sample batch of data
    # Input:
    #    dataset: (data, trajectory_starts, policy_starts)
    #    data: [n_tasks, n_sample, dim]
    #    tasks: list of task id, if None, sample from all train tasks
    # Output:
    #    sampled_data: [n_tasks, batch_size, context_len, dim]
    #    tasks: list of task id. 
    ds, ts, ps = dataset
    n_tasks = len(ds)
    # n_tasks, n_samples, _ = ds.shape
    tasks = tasks if tasks is not None else np.arange(n_tasks)
    sampled_data = []
    percentile = percentile or [0,1]
    for t in tasks:
        task_data = ds[t]
        n_samples, _ = task_data.shape
        traj_split = ts[t]
        lo = int(n_samples * percentile[0])
        hi = int(n_samples * percentile[1])
        query_index = np.random.randint(lo, hi+1-context_len, size = (batchsize))
        # query_index = np.random.randint(0, n_samples+1-context_len, size=(batchsize))
        t_data = []
        for q in query_index:
            while not np.all((q < traj_split) == (q+context_len<traj_split)):
                q = np.random.randint(0, n_samples+1-context_len)
            t_data.append( task_data[q:q+context_len, :] )
        # t_data = ds[t, query_index]
        sampled_data.append(t_data)
    sampled_data = np.array(sampled_data)
    
    if arr_type == 'torch':
        device = ptu.device
        sampled_data = ptu.FloatTensor(sampled_data).to(device)
        
    return sampled_data, tasks
    
    
def locate_index(value, split, max_length):
    ns = np.concatenate([split, [max_length]])
    low = ns[value>=ns][-1]
    high = ns[value<ns][0]
    return low, high

def is_same_split(a, b, split):
    return np.all((a<split) == (b<split))

def verified_sample(split, context_len=1, lo=0, hi=100, n_samples=1000):
    ind = np.random.randint(lo, hi-context_len)
    tries = 0
    low, high = locate_index(ind, split, n_samples)
    while high-low<=context_len or not np.all((ind<split) == (ind+context_len-1<split)):
        ind = np.random.randint(lo, hi-context_len)
        low, high = locate_index(ind, split, n_samples)
        tries += 1
        if tries >= 100: raise NotImplementedError
    return ind

def sample_pos_neg_batch_data(dataset, batchsize, context_len = 1, n_layer = 1, split_type = 'task', arr_type='torch'):
    
    ds, ts, _ = dataset
    n_tasks = len(ds)
    task_index = np.random.randint(0, n_tasks, size=(batchsize))
    sampled_data = []
    for i in range(batchsize):
        t_ind = task_index[i]
        task_data = ds[t_ind]
        traj_split=ts[t_ind]
        
        n_samples, _ = task_data.shape
        
        q_ind = verified_sample(traj_split, context_len, lo=0, hi=n_samples, n_samples=n_samples)
        # lo, hi = locate_index(q_ind, traj_split, n_samples)
        # if split_type == 'task':
        #     lo, hi = 0, n_samples
        # elif split_type == 'policy':
        #     lo, hi = locate_index(q_ind, policy_split, n_samples)
        # elif split_type == 'traj':
        lo, hi = locate_index(q_ind, traj_split, n_samples)
        k_ind = verified_sample(traj_split, context_len, lo=lo, hi=hi, n_samples=n_samples)
        query_batch_data = [task_data[q_ind:q_ind+context_len, :],
                            task_data[k_ind:k_ind+context_len, :]]

        view_data =  np.stack( query_batch_data, axis = 0 )
        sampled_data.append(view_data)
        
    sampled_data = np.asarray(sampled_data)
    

    if split_type == 'SupCL':
        task_index = np.reshape(task_index, (-1,1))
        task_label_mask = np.equal(task_index, task_index.T)
        task_label_mask = np.asarray(task_label_mask, dtype='uint8')
    elif split_type == 'SelfCL': 
        task_label_mask = np.eye(batchsize, dtype='uint8')
    else:
        raise NotImplementedError
        
    if arr_type=='torch':
        device = ptu.device
        sampled_data = ptu.FloatTensor(sampled_data).to(device)
        task_label_mask = ptu.FloatTensor(task_label_mask).to(device)
        
    
    return sampled_data, task_label_mask
    

def sample_pos_neg_batch_data_old(dataset, batchsize, context_len = 1, n_layer = 3, split_type = 'task',  arr_type='torch'):
    # sample batch of contrastive data:
    # Input: 
    #     dataset: (data, trajectory_starts, policy_starts)
    #     data: [n_tasks, n_sample, sum(dim)]
    #     trajectory_starts, policy_starts: [n_tasks, num_traj/num_policy]
    #     
    # Output:
    #     sampled_data: [batchsize, n_views (typically 2), context_len, dim]
    #     masks: [n_layers, batchsize, batchsize]
    #     n_layer: 
    #         1 - tasks (batchsize, n_views, ...), 
    #         2 - policies (batchsize * 2policies, n_views, ...)
    #         3 - trajectories (batchsize *2policies *2trajectories, n_views, ...)
    
    ds, ts, ps = dataset
    n_tasks = len(ds)
    # n_samples, _ = ds.shape
    task_index = np.random.randint(0, n_tasks, size=(batchsize))
    # query_index = np.random.randint(0, n_samples-context_len, size=(batchsize))
    # key_index = np.random.randint(0, n_samples-context_len, size=(batchsize))
    
    sampled_data = []
    loopsize = int(batchsize / 2**(n_layer-1))
    for i in range(loopsize):
        t_ind = task_index[i]
        task_data = ds[t_ind]
        traj_split = ts[t_ind]
        policy_split = ps[t_ind]
        n_samples, _ = task_data.shape
        # q_ind = query_index[i]
        # k_ind = key_index[i]
        # query_batch_data = []
        
        if n_layer == 3:
            
            # sample each layer of data separately, 
            # sample has different label with the lower layer, but the same lable from the current layer
            
            # sample the first layer data (trajectory level): (batchsize * 2 * dim)
            l1_q_ind = verified_sample(traj_split, context_len, lo=0, hi=n_samples, n_samples=n_samples)
            lo, hi = locate_index(l1_q_ind, traj_split, n_samples)
            # while hi - lo < context_len:
            #     l1_q_ind = verified_sample(traj_split, context_len, lo=0, hi=n_samples)
            #     lo, hi = locate_index(l1_q_ind, traj_split, n_samples)
            k_ind = verified_sample(traj_split, context_len, lo=lo, hi=hi, n_samples=n_samples)
            query_batch_data = [task_data[l1_q_ind:l1_q_ind+context_len, :],
                                task_data[k_ind:k_ind+context_len, :]]
            view_data =  np.stack( query_batch_data, axis = 0 )
            sampled_data.append(view_data)
            
            
            # # sample the second layer data (policy level), : (batchsize * 2 * dim)
            lo, hi = locate_index(l1_q_ind, policy_split, n_samples)
            tries = 0
            l2_q_ind = verified_sample(traj_split, context_len, lo=lo, hi=hi, n_samples=n_samples)
            # lo, hi = locate_index(l2_q_ind, traj_split, n_samples)
            while is_same_split(l1_q_ind, l2_q_ind, traj_split):
                tries += 1
                l2_q_ind = verified_sample(traj_split, context_len, lo=lo, hi=hi, n_samples=n_samples)
                # lo, hi = locate_index(l2_q_ind, traj_split, n_samples)
                if tries > 100: raise NotImplementedError
            lo, hi = locate_index(l2_q_ind, traj_split, n_samples)
            k_ind = verified_sample(traj_split, context_len, lo=lo, hi=hi, n_samples=n_samples)
            l2_query_batch_data = [task_data[l2_q_ind:l2_q_ind+context_len, :],
                                   task_data[k_ind:k_ind+context_len, :]]
            view_data =  np.stack( l2_query_batch_data, axis = 0 )
            sampled_data.append(view_data)
            
            
            # sample the third layer data (task level)
            lo, hi = 0, n_samples
            tries = 0
            l3_q_ind = verified_sample(traj_split, context_len, lo=lo, hi=hi, n_samples=n_samples)
            lo, hi = locate_index(l3_q_ind, traj_split, n_samples)
            while is_same_split(l2_q_ind, l3_q_ind, policy_split):
                tries += 1
                l3_q_ind = verified_sample(traj_split, context_len, lo=lo, hi=hi)
                if tries > 100: raise NotImplementedError
            lo, hi = locate_index(l3_q_ind, traj_split, n_samples)
            k_ind = verified_sample(traj_split, context_len, lo=lo, hi=hi, n_samples=n_samples)
            l3_query_batch_data = [task_data[l3_q_ind:l3_q_ind+context_len, :],
                                   task_data[k_ind:k_ind+context_len, :]]
            view_data =  np.stack( l3_query_batch_data, axis = 0 )
            sampled_data.append(view_data)
            
            
            # sample the fourth, very suboptimal!!!
            lo, hi = locate_index(l3_q_ind, policy_split, n_samples)
            tries = 0
            l4_q_ind = verified_sample(traj_split, context_len, lo=lo, hi=hi, n_samples=n_samples)
            while is_same_split(l3_q_ind, l4_q_ind, traj_split):
                tries += 1
                l4_q_ind = verified_sample(traj_split, context_len, lo=lo, hi=hi, n_samples=n_samples)
                if tries > 100: raise NotImplementedError
            lo, hi = locate_index(l4_q_ind, traj_split, n_samples)
            k_ind = verified_sample(traj_split, context_len, lo=lo, hi=hi, n_samples=n_samples)
            l4_query_batch_data = [task_data[l4_q_ind:l4_q_ind+context_len, :],
                                   task_data[k_ind:k_ind+context_len, :]]
            view_data = np.stack( l4_query_batch_data, axis = 0 )
            sampled_data.append(view_data)
            
            
        elif n_layer == 2:
            l1_q_ind = verified_sample(traj_split, context_len, lo=0, hi=n_samples, n_samples=n_samples)
            lo, hi = locate_index(l1_q_ind, traj_split, n_samples)
            # while hi - lo < context_len:
            #     l1_q_ind = verified_sample(traj_split, context_len, lo=0, hi=n_samples)
            #     lo, hi = locate_index(l1_q_ind, traj_split, n_samples)
            k_ind = verified_sample(traj_split, context_len, lo=lo, hi=hi, n_samples=n_samples)
            query_batch_data = [task_data[l1_q_ind:l1_q_ind+context_len, :],
                                task_data[k_ind:k_ind+context_len, :]]
            view_data =  np.stack( query_batch_data, axis = 0 )
            sampled_data.append(view_data)
            
            lo, hi = 0, n_samples
            tries = 0
            l3_q_ind = verified_sample(traj_split, context_len, lo=lo, hi=hi, n_samples=n_samples)
            # lo, hi = locate_index(l3_q_ind, traj_split, n_samples)
            while is_same_split(l1_q_ind, l3_q_ind, traj_split):
                tries += 1
                l3_q_ind = verified_sample(traj_split, context_len, lo=lo, hi=hi, n_samples=n_samples)
                if tries > 100: raise NotImplementedError
            lo, hi = locate_index(l3_q_ind, traj_split, n_samples)
            k_ind = verified_sample(traj_split, context_len, lo=lo, hi=hi, n_samples=n_samples)
            l3_query_batch_data = [task_data[l3_q_ind:l3_q_ind+context_len, :],
                                   task_data[k_ind:k_ind+context_len, :]]
            view_data =  np.stack( l3_query_batch_data, axis = 0 )
            sampled_data.append(view_data)
            
        
        elif n_layer == 1:
            
            q_ind = verified_sample(traj_split, context_len, lo=0, hi=n_samples, n_samples=n_samples)
            # lo, hi = locate_index(q_ind, traj_split, n_samples)
            if split_type == 'task':
                lo, hi = 0, n_samples
            elif split_type == 'policy':
                lo, hi = locate_index(q_ind, policy_split, n_samples)
            elif split_type == 'traj':
                lo, hi = locate_index(q_ind, traj_split, n_samples)
            k_ind = verified_sample(traj_split, context_len, lo=lo, hi=hi, n_samples=n_samples)
            query_batch_data = [task_data[q_ind:q_ind+context_len, :],
                                task_data[k_ind:k_ind+context_len, :]]
        
            view_data =  np.stack( query_batch_data, axis = 0 )
            sampled_data.append(view_data)
    
    sampled_data = np.asarray(sampled_data)
    if n_layer == 3:
        
        # overall_size = sampled_data.shape[0]
        # print(batchsize, sampled_data.shape)
        assert batchsize == sampled_data.shape[0]
        mask1 = np.eye(batchsize, dtype='uint8')
        ones_matrix = np.ones([2,2], dtype = 'uint8')
        mask2 = block_diag(*[ones_matrix]*(loopsize*2))
        ones_matrix = np.ones([4,4], dtype='uint8')
        mask3 = block_diag(*[ones_matrix]*loopsize)
        task_label_mask = np.asarray([mask3, mask2, mask1])
    
    elif n_layer == 2:
        assert batchsize == sampled_data.shape[0]
        mask1 = np.eye(batchsize, dtype='uint8')
        ones_matrix = np.ones([2,2], dtype = 'uint8')
        mask2 = block_diag(*[ones_matrix]*(loopsize))
        task_label_mask = np.asarray([mask2, mask1])
        
    elif n_layer == 1:
        task_label_mask = np.eye(batchsize, dtype='uint8')
    # tmp = np.repeat([task_index], batchsize, axis = 0)
    # task_label_mask = np.array( tmp == tmp.T, dtype = 'uint8')
    if arr_type=='torch':
        device = ptu.device
        sampled_data = ptu.FloatTensor(sampled_data).to(device)
        task_label_mask = ptu.FloatTensor(task_label_mask).to(device)
        
    
    return sampled_data, task_label_mask
        
    

    
def sample_pos_neg_batch_data_new(dataset, batchsize, context_len = 1, n_layer = 3, split_type = 'task',  arr_type='torch'):
    

    ds, ts, ps = dataset
    n_tasks = len(ds)
    # n_samples, _ = ds.shape
    task_index = np.random.randint(0, n_tasks, size=(batchsize))
    # query_index = np.random.randint(0, n_samples-context_len, size=(batchsize))
    # key_index = np.random.randint(0, n_samples-context_len, size=(batchsize))
    
    sampled_data = []
    
    for i in range(batchsize):
        t_ind = task_index[i]
        task_data = ds[t_ind]
        traj_split = ts[t_ind]
        policy_split = ps[t_ind]
        n_samples, _ = task_data.shape
        
        q_ind = verified_sample(traj_split, context_len, lo=0, hi=n_samples, n_samples=n_samples)
        
        
def sample_batch_data_new(dataset, batchsize, context_len = 200, split_type='traj'):
    ds, ts, ps = dataset
    
    n_tasks = len(ds)
    
    task_index = np.random.randint(0, n_tasks, size = (batchsize))
    sampled_data = []
    
    for i in range(batchsize):
        t_ind = task_index[i]
        task_data = ds[t_ind]
        traj_split = ts[t_ind]
        n_samples, _ = task_data.shape

        q_ind = verified_sample(traj_split, context_len=1, lo=0, hi=n_samples, n_samples=n_samples)
        lo, hi = locate_index(q_ind, traj_split, n_samples)
        
        view_data = task_data[lo:hi, :]
        
        sampled_data.append(view_data)
    
    return sampled_data


# def augment_batch_data(data, augmentation='mask'):
#     sampled_data = []
#     for d in data:
        