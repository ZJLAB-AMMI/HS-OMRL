import argparse
import torch
from utils.cli import boolean_argument


def get_args(rest_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='PointRobot-v1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--policy', default='sac')
    parser.add_argument('--max-rollouts-per-task', default=1) # should be 1, not BAMDP
    parser.add_argument('--policy-buffer-size', default=1e6)
    parser.add_argument('--dqn-layers', default=[32, 32])
    parser.add_argument('--policy-layers', default=[32, 32])
    parser.add_argument('--actor-lr', default=3e-4)
    parser.add_argument('--critic-lr', default=3e-4)
    parser.add_argument('--entropy-alpha', default=0.01)
    parser.add_argument('--automatic-entropy-tuning', default=False)
    parser.add_argument('--alpha-lr', default=None)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--soft-target-tau', default=0.005)
    parser.add_argument('--num-iters', default=200, type=int)
    parser.add_argument('--num-init-rollouts-pool', default=10)
    parser.add_argument('--num-rollouts-per-iter', default=2)
    parser.add_argument('--batch-size', default=256)
    parser.add_argument('--rl-updates-per-iter', default=10, type=int)
    parser.add_argument('--eval-deterministic', default=True)
    parser.add_argument('--layer-norm', default=False)
    parser.add_argument('--batch-norm', default=False)

    # Ablation - initial state distribution
    parser.add_argument('--modify-init-state-dist', default=False)
    
    parser.add_argument('--log-interval', default=10)
    parser.add_argument('--save-interval', default=20)

    parser.add_argument('--save-buffer', default=True, type=int)  # TODO - if false, crashes
    parser.add_argument('--save-models', default=False, type=int)
    parser.add_argument('--main-save-dir', default='./batch_data')
    parser.add_argument('--log-tensorboard', default=False, type=int)
    parser.add_argument('--save-dir', default='data')
    parser.add_argument('--agent-log-dir', default='tmp/gym/', help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--results-log-dir', default=None, help='directory to save agent logs (default: ./data)')

    args = parser.parse_args(rest_args)

    return args
