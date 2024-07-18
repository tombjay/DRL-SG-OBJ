"""
This file consists the code for evaluating object-centric models.
Change the path of saved model in 'model_file' variable 
and change the game environment in 'env_id' variable.
"""


import os
import random
import time
from dataclasses import dataclass
import cv2

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import matplotlib.pyplot as plt
from torch.distributions.categorical import Categorical
from slot_attention import SlotAttentionClassifier

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 2
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    env_id: str = "ALE/Freeway-v5"
    """the id of the environment"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Thesis-RL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    model_path: str = "/scratch/users/sundararaj/msc2023_jayakumar/fasterRCNN/sgg/checkpoints/clean_ppo"
    """directory to save the model"""
    num_envs: int = 8
    """the number of parallel game environments"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class SlotAttentionAgent(nn.Module):
    def __init__(self, envs, input_size = 64, hidden_size = 512):
        super(SlotAttentionAgent, self).__init__()
        self.slot_attention = SlotAttentionClassifier()
        
        self.slot_processing_layer = nn.Sequential(
            layer_init(nn.Linear(input_size, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 256)),
            nn.ReLU(),
        )
        
        self.actor = layer_init(nn.Linear(256, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)
        
    def forward(self, obs):
        slot_outputs = []
        slots = self.slot_attention(obs.to(device))
        for slot in slots.permute(1,0,2):  # Iterate over slots
            slot_output = self.slot_processing_layer(slot)
            slot_outputs.append(slot_output)
        slot_outputs = torch.stack(slot_outputs)  # Stack slot outputs along slot dimension
        aggregated_output = torch.mean(slot_outputs, dim=0)  # Mean pooling
        return aggregated_output
    
    def get_value(self, unprocessed_obs):
        hidden_value = self.forward(unprocessed_obs.to(device))
        critic_value = self.critic(hidden_value.to(device))
        return critic_value

    def get_action_and_value(self, unprocessed_obs, action = None):
        hidden_value = self.forward(unprocessed_obs.to(device))
        logits = self.actor(hidden_value.to(device))
        critic_value = self.critic(hidden_value.to(device))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), critic_value


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"/scratch/users/sundararaj/msc2023_jayakumar/fasterRCNN/sgg/videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"/scratch/users/sundararaj/msc2023_jayakumar/fasterRCNN/sgg/videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (128, 128))
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def evaluate_model(eval_env, agent):
    episode_return = [] 
    episodic_returns = [[] for _ in range(eval_env.num_envs)]  # List to store returns for each environment
    mean_episodic_returns = []
    for r in range(1000):  # Perform 10 evaluation episodes
        obs, _ = eval_env.reset()
        ep_return = 0
        all_done = [False] * len(obs)  # Initialize list of boolean values for each environment
        while True:  # Continue until all environments are done
            obs = torch.Tensor(obs).to(device)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs)
            obs, reward, done, truncations, infos = eval_env.step(action.cpu().numpy())
            for i in range(len(done)):
                all_done[i] = all_done[i] or done[i]  # Update all_done flag for each environment
            if all(all_done):
                break  # Break out of the loop if all environments are done
            ep_return += reward

        episode_return.append(np.mean(ep_return))
        
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    mean_episodic_returns.append(info['episode']['r'])

    mean_return = np.mean(mean_episodic_returns)
    std_return = np.std(mean_episodic_returns)
    return mean_return, std_return

if __name__ == "__main__":
    args = tyro.cli(Args)
    model_file = os.path.join(args.model_path, 'OBJ_SLOT_FW_10M')
    run_name = f"{args.env_id}_evaluate_OBJ_SLOT_FB_10M{args.exp_name}__{args.seed}__{int(time.time())}"
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    
    if args.track:
        import wandb
        wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
            )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"


    agent = SlotAttentionAgent(envs).to(device)
    agent.eval()
    
    print("Loading model from:", model_file)
    agent.load_state_dict(torch.load(model_file))
    
    if isinstance(agent, SlotAttentionAgent):
        num_learnable_params = 0
        num_non_learnable_params = 0

        # Iterate through the parameters of the model
        for param in agent.parameters():
            if param.requires_grad:
                num_learnable_params += param.numel()
            else:
                num_non_learnable_params += param.numel()

        # Print the counts
        print("Number of Learnable Parameters:", num_learnable_params)
        print("Number of Non-Learnable Parameters:", num_non_learnable_params)

    else:
        print("Loaded model is not an instance of torch.nn.Module.")
    
    mean_reward, std_dev = evaluate_model(envs, agent)
    print('Evaluation rewards', mean_reward, 'Std.deviation', std_dev)