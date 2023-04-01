## On Context Distribution Shift in Task Representation Learning for Offline Meta RL

### Abstract
Offline meta reinforcement learning (OMRL) aims to learn transferrable knowledge from offline datasets to facilitate the learning process for new target tasks. Context-based RL employs a context encoder to rapidly adapt the agent to new tasks by inferring about the task representation, and then adjusting the acting policy based on the inferred task representation. Here we consider context-based OMRL, in particular, the issue of task representation learning for OMRL. We empirically demonstrate that the context encoder trained on offline datasets could suffer from distribution shift between the contexts used for training and testing. To tackle this issue, we propose a hard sampling based strategy for learning a robust task context encoder. Experimental results, based on distinct continuous control tasks, demonstrate that the utilization of our technique results in more robust task representations and better testing performance in terms of accumulated returns, compared with baseline methods.

### Experiments

We demonstrate with PointRobotGoal environment. For other environments, change the argument `--env-type` according to the table:

Environment  | Argument
------------- | -------------
Point-Robot  | point_goal
Half-Cheetah-Vel  | cheetah_vel
Ant-Dir | ant_dir
Hopper-Param | hopper_param
Walker-Param | walker_param

#### Data Collection
Copy the following code into a shell script, and run the script.
```
python train_data_collection.py --env-type point_goal --save-models 1 --log-tensorboard 1 --seed $seed
```

#### Train the Task Encoder
```
python train_contrastive.py --env-type point_goal --layer-type SupCL --contrastive-loss combine
```
To train with different sampling strategies, replace `combine` with `hard_neg`, `hard_pos` or `easy`.

#### Offline Meta-RL
```
python train_offpolicy_with_trained_encoder.py --env-type point_goal  --encoder-model-path PATH_TO_MODEL
```
#### Acknowledgement
This code is based on [CORRO](https://github.com/PKU-RL/CORRO).
