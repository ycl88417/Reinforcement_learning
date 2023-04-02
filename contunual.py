import setup_path
import gym
import airgym
import time

import torch as th
import torch.nn as nn
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Callable


# Create a DummyVecEnv for main airsim gym env
env = DummyVecEnv(
    [
        lambda: Monitor(
            gym.make(
                "airgym:airsim-drone-sample-v0",
                ip_address="127.0.0.1",
                step_length=3,
                image_shape=(84, 84, 3),
            )
        )
    ]
)

# Wrap env as VecTransposeImage to allow SB to handle frame observations
env = VecTransposeImage(env)

#policy_kwargs = dict(activation_fn=th.nn.ReLU,#)
step = '40000'
models_dir = "."
model_path = f"{models_dir}\\{step}"

model = DQN.load(model_path,env=env,tensorboard_log=".\\tb_logs")
print(model)
# Create an evaluation callback with the same env, called every 5000 iterations
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path=".",
    log_path=".",
    eval_freq=5000,
)
callbacks.append(eval_callback)
kwargs = {}
kwargs["callback"] = callbacks

timesteps = 10000

for i in range(50):
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False,**kwargs)
    model.save(f"{models_dir}/{40000+timesteps*i}")




