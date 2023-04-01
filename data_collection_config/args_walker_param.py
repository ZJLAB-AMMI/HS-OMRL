import argparse
import torch
from utils.cli import boolean_argument

# official sac: 1e6 steps, every step rollout 1 transition, update 1 step

def get_args(rest_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='WalkerRandParams-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--policy', default='sac')
    parser.add_argument('--max-rollouts-per-task', default=1) # should be 1, not BAMDP
    parser.add_argument('--policy-buffer-size', default=1e6)
    parser.add_argument('--dqn-layers', default=[128, 128])
    parser.add_argument('--policy-layers', default=[128, 128])
    parser.add_argument('--actor-lr', default=3e-4)
    parser.add_argument('--critic-lr', default=3e-4)
    parser.add_argument('--entropy-alpha', default=0.2)
    parser.add_argument('--automatic-entropy-tuning', default=False)
    parser.add_argument('--alpha-lr', default=None)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--soft-target-tau', default=0.005)
    parser.add_argument('--num-iters', default=3000, type=int)
    parser.add_argument('--num-init-rollouts-pool', type=int, default=50)
    parser.add_argument('--num-rollouts-per-iter', default=2)
    parser.add_argument('--batch-size', default=256)
    parser.add_argument('--rl-updates-per-iter', default=200)
    parser.add_argument('--eval-deterministic', default=True)

    # Ablation - initial state distribution
    parser.add_argument('--modify-init-state-dist', default=False)

    parser.add_argument('--log-interval', default=10)
    parser.add_argument('--save-interval', default=20)

    parser.add_argument('--save-buffer', type=int, default=True)  # TODO - if false, crashes
    parser.add_argument('--save-models', type=int, default=False)
    parser.add_argument('--main-save-dir', default='./batch_data')
    parser.add_argument('--log-tensorboard', type=int, default=False)
    parser.add_argument('--save-dir', default='data')
    parser.add_argument('--agent-log-dir', default='tmp/gym/', help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--results-log-dir', default=None, help='directory to save agent logs (default: ./data)')

    args = parser.parse_args(rest_args)

    return args
