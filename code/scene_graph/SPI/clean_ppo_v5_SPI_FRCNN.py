"""
-->This file consists the code for training GNN agent using GCNConv model 
and node features obtained from FasterRCNN model.
"""
import os
import random
import time
from dataclasses import dataclass
import cv2

import gymnasium as gym
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn.conv import GATv2Conv, GCNConv
from torch_geometric.nn import aggr, Sequential as Seq
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.nn.pool import global_mean_pool, global_max_pool
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

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
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Thesis-RL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """Whether to save model (check out 'checkpoints' folder)"""
    save_interval: int = 1000
    """the interval at which to save the model during training"""
    model_path: str = "/scratch/users/sundararaj/msc2023_jayakumar/fasterRCNN/sgg/checkpoints/clean_ppo"
    """directory to save the model"""
    evaluate: bool = False
    """if toggled, model will be evaluated during the training process"""
    evaluation_interval: int = 10
    """the interval at which the model will be evaluated"""

    # Algorithm specific arguments
    env_id: str = "ALE/SpaceInvaders-v5"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

# Function to load FasterRCNN model.
def get_object_detection_model(num_classes, model_path = None):

    fasterrcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = fasterrcnn_model.roi_heads.box_predictor.cls_score.in_features
    fasterrcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load saved weights if a model_path is provided
    if model_path:
        loaded_model_state = torch.load(model_path)
        fasterrcnn_model.load_state_dict(loaded_model_state)

    return fasterrcnn_model

# Code to load saved FasterRCNN model.
def get_object_features():
    fasterrcnn_model_path = r'/scratch/users/sundararaj/msc2023_jayakumar/fasterRCNN/sgg/fasterRCNN-SPI-10.pth'
    fasterrcnn_model = get_object_detection_model(num_classes = 8, model_path = fasterrcnn_model_path)
    feature_outputs = []
    sample_image_path = r'/scratch/users/sundararaj/msc2023_jayakumar/fasterRCNN/sgg/RZ_3972461_4451.png'
    width = 160
    height = 210
    img = cv2.imread(sample_image_path)
    img_res = cv2.resize(img, (width, height), cv2.INTER_AREA)
    img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
    img_res = img_res / 255.0  # Normalizing pixel values

    process_image = torch.tensor(img_res, dtype=torch.float32).permute(2, 0, 1)  # Channels-first
    
    fasterrcnn_model.eval()
    hook = fasterrcnn_model.roi_heads.box_predictor.cls_score.register_forward_hook(
        lambda layer, input, output: feature_outputs.append(output))

    res = fasterrcnn_model([process_image])
    hook.remove()
    proposal_boxes = feature_outputs[0]
    return proposal_boxes


# Global FasterRCNN object features.
faster_rcnn_features = get_object_features()

def make_env(env_id, idx, capture_video, run_name):
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
        return env

    return thunk


def evaluate_model(eval_env, agent):
    episodic_returns = [[] for _ in range(eval_env.num_envs)]  # List to store returns for each environment
    mean_episodic_returns = []
    for r in range(10):  # Perform 10 evaluation episodes
        obs, _ = eval_env.reset()
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
            
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    mean_episodic_returns.append(info['episode']['r'])

    mean_return = np.mean(mean_episodic_returns)
    return mean_return


def generate_bounding_boxes(frames):
    batch_boxes = []
    batch_labels = []
    frames = frames.cpu().numpy()
    for frame in frames:
        boxes, labels = [], []

        for obj_class, color in object_colors.items():
            minx, miny, maxx, maxy, closing_dist = get_object_coordinates(obj_class)

            masks = [cv2.inRange(frame[miny:maxy, minx:maxx, :], np.array(color), np.array(color)) for color in color]
            mask = sum(masks)

            closed = closing(mask, square(closing_dist))

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x, y = x + minx, y + miny
                boxes.append([x, y, x + w, y + h])
                labels.append(object_id_mapping[obj_class])

        filtered_boxes = [box for box, label in zip(boxes, labels) if label not in labels_to_remove]
        filtered_labels = [label for label in labels if label not in labels_to_remove]

        
        batch_boxes.append(filtered_boxes)
        batch_labels.append(filtered_labels)

    return batch_boxes, batch_labels


object_colors = {
    "player": [[50, 132, 50]],
    "score": [[50, 132, 50]],
    "alien": [[134, 134, 29]],
    "shield": [[181, 83, 40]],
    "satellite": [[151, 25, 122]],
    "bullet": [[142, 142, 142]],
    "lives": [[162, 134, 56]]
}

object_id_mapping = {
    "background": 0,
    "player": 1,
    "score": 2, 
    "alien": 3, 
    "shield": 4, 
    "satellite": 5, 
    "bullet": 6, 
    "lives": 7
}

# Remove unwanted objects in the frame.
labels_to_remove = [0, 2, 7]

def get_object_coordinates(obj_class):
    minx, miny, maxx, maxy, closing_dist = 0, 0, 160, 210, 3

    if obj_class == "player":
        miny, maxy = 180, 195
    elif obj_class == "score":
        maxy, closing_dist = 30, 12
    
    return minx, miny, maxx, maxy, closing_dist

def closing(mask, kernel):
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def square(size):
    return np.ones((size, size), dtype=np.uint8)

def vector_encoding(frames):
    batch_boxes, batch_labels = generate_bounding_boxes(frames)
    max_objects_SPI = 45 # To get a fixed length of vector.
    max_fixed_length = max_objects_SPI * faster_rcnn_features.shape[0]
    
    batched_object_features = []
    
    for labels in zip(batch_labels):
        object_features = []
        
        for label in enumerate(labels):
            object_features.append(faster_rcnn_features[label])
        
        single_env_obj_feature = torch.cat(object_features, dim = 0)
        
        if single_env_obj_feature.shape[0] < max_fixed_length:
            padding = torch.zeros(max_fixed_length - single_env_obj_feature.shape[0], dtype=single_env_obj_feature.dtype)
            single_env_obj_feature = torch.cat([single_env_obj_feature, padding])
        elif concatenated_features.shape[0] > max_fixed_length:
            # Truncate the features to match the fixed length
            single_env_obj_feature = single_env_obj_feature[:max_fixed_length]
            
        batched_object_features.append(single_env_obj_feature)
        
    batched_object_features = torch.stack(batched_object_features, dim=0)
    print(batched_object_features.shape)

    return batched_object_features


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, input_dim = 45000, hidden_dim = 4096):
        super().__init__()
        
        self.fc_network = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 2048)),
            nn.ReLU(),
            layer_init(nn.Linear(2048, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.ReLU(),
        )
        
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def forward(self, encoded_vector):

        hidden = self.fc_network(encoded_vector)
        return hidden
    
    def get_value(self, unprocessed_obs):
        
        encoded_vector = vector_encoding(unprocessed_obs)
        batch_outputs = self.forward(encoded_vector.to(device))
        critic_value = self.critic(batch_outputs.to(device))
        return critic_value

    def get_action_and_value(self, unprocessed_obs, action = None):
        
        encoded_vector = vector_encoding(unprocessed_obs)
        hidden = self.forward(encoded_vector.to(device))
        logits = self.actor(hidden)
        critic_value = self.critic(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), critic_value


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    model_file = os.path.join(args.model_path, 'Obj_FRCNN_v7_10M')
    run_name = f"{args.env_id}__Obj_FRCNN_v7_10M{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            # save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    print(agent)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    if isinstance(agent, Agent):
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

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        
        if args.save_model and iteration % args.save_interval == 0:
            torch.save(agent.state_dict(), model_file)
            print(f"Model saved at iteration {iteration}.")
        
        if args.evaluate and iteration % args.evaluation_interval == 0:
            performance = evaluate_model(envs, agent)
            print('Evaluation rewards', performance)
            writer.add_scalar("evaluation/episodic_return", performance, iteration)
        
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            writer.add_histogram('histograms/action_histogram', actions, global_step)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()