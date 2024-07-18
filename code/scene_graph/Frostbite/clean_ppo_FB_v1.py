"""
This file consists the code for training GNN agent using GATv2Conv model 
and with unprocessed node embeddings of size 5.

"""
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import time
import cv2
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn.conv import GATv2Conv, GCNConv
from torch_geometric.nn import aggr, Sequential as Seq
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.nn.pool import global_mean_pool, global_max_pool

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
    env_id: str = "ALE/Frostbite-v5"
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

    return thunk

def generate_bounding_boxes(frames):
    batch_boxes = []
    batch_labels = []
    frames = frames.cpu().numpy()
    for frame in frames:
        boxes, labels = [], []

        for obj_class, color in objects_color.items():
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

def create_graph_input(frames):

    batch_boxes, batch_labels = generate_bounding_boxes(frames)

    batch_node_features = []
    batch_edge_index = []

    for boxes, labels in zip(batch_boxes, batch_labels):
        node_features = []
        
        for i, label in enumerate(labels):
            node_feature = torch.tensor(boxes[i])

            # Encode whether the node is friendly or not
            if label in [1, 2, 3, 5, 6]:
                friendly_indicator = torch.tensor([1.])  # Friendly node
            else:
                friendly_indicator = torch.tensor([0.])  # Not friendly node

            # Concatenate bounding box coordinates, label encoding, and friendly indicator
            node_feature = torch.cat((node_feature, friendly_indicator))

            node_features.append(node_feature)

        
        node_features = torch.stack(node_features)


        # Compute pairwise distances between all pairs of nodes
        num_nodes = node_features.size(0)

        # bounding boxes are represented by node_features[:, :4]
        bounding_boxes = node_features[:, :4]

        # Compute centroids of bounding boxes
        centroids = (bounding_boxes[:, :2] + bounding_boxes[:, 2:]) / 2  # Shape: (num_nodes, 2)

        # Compute pairwise differences between centroids
        diffs = centroids.unsqueeze(1) - centroids.unsqueeze(0)  # Shape: (num_nodes, num_nodes, 2)

        # Compute pairwise distances along the last dimension (Euclidean distance)
        pairwise_distances = torch.norm(diffs, dim=-1)  # Shape: (num_nodes, num_nodes)

        # Set distance threshold
        distance_threshold = 20

        # Create mask for nodes with label 1
        labels_tensor = torch.tensor(labels)
        label_1_mask = labels_tensor == 1

        # Create mask for nodes based on distance threshold
        distance_mask = pairwise_distances < distance_threshold

        # Create adjacency matrix
        adjacency_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.int)

        # Connect nodes with label 1 to all other nodes
        adjacency_matrix[label_1_mask, :] = 1

        # Connect nodes based on distance threshold
        adjacency_matrix[~label_1_mask & distance_mask] = 1

        # Convert adjacency matrix to edge index format
        edge_index = adjacency_matrix.nonzero(as_tuple=False).t()

        batch_node_features.append(node_features)
        batch_edge_index.append(edge_index)

    batch_data_list = [Data(x=node_features, edge_index=edge_index) for node_features, edge_index in zip(batch_node_features, batch_edge_index)]
    batch = Batch.from_data_list(batch_data_list)

    return batch

# Define object colors and mappings
objects_color = {
    "WhitePlate": [[214,214,214]], 
    "BluePlate": [[84,138,210]],
    "Bird": [[132,144,252]],
    "hud_objs": [[132,144,252]],
    "house": [[142,142,142],[0,0,0]],
    "greenfish": [[111,210,111]],
    "crab": [[213,130,74]], 
    "clam": [[210,210,64]], 
    "bear": [[111,111,111]], 
    "player": [[162, 98,33], [198,108,58], [142,142,142],[162,162,42]] 
}

object_id_mapping = {
    "background": 0,
    "player": 1,
    "WhitePlate": 2,
    "BluePlate": 3,
    "Bird": 4,
    "house": 5,
    "greenfish": 6,
    "crab": 7,
    "clam": 8,
    "bear": 9,
    "frostbite": 10,
    "hud_objs": 11
}

labels_to_remove = [11] # Remove unwanted objects in the frame.

def get_object_coordinates(obj_class):
    # Default coordinates
    minx, miny, maxx, maxy, closing_dist = 0, 0, 160, 210, 3

    # Adjust coordinates based on object class
    if obj_class == "WhitePlate":
        maxy = 185
    elif obj_class == "player":
        miny = 60
    elif obj_class == "house":
        minx, miny, maxx, maxy, closing_dist = 84, 13, 155, 64, 1
    elif obj_class == "Bird":
        miny, maxy, closing_dist = 75, 180, 5
    elif obj_class == "bear":
        miny, maxy = 13, 75
    elif obj_class in ["crab", "clam", "frostbite"]:
        miny, maxy = 75, 180
    
    return minx, miny, maxx, maxy, closing_dist

def closing(mask, kernel):
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def square(size):
    return np.ones((size, size), dtype=np.uint8)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

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



class Agent(nn.Module):
    def __init__(self, envs, input_node_dim = 5, hidden_dim = 128, gcn_layers = 2):
        super().__init__()
        
        self.GATv2Conv = Seq('x, edge_index', [
            (GATv2Conv(in_channels = input_node_dim, out_channels = hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GATv2Conv(in_channels = hidden_dim, out_channels = hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GATv2Conv(in_channels = hidden_dim, out_channels = hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(inplace=True)
        ])
        
        self.fc_network = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 32)),
            nn.ReLU(),
            layer_init(nn.Linear(32, 16)),
            nn.ReLU(),
        )
        
        self.actor = layer_init(nn.Linear(16, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(16, 1), std=1)

    def forward(self, graph_data, num_graphs, ptr):
        
        gcn_output = self.GATv2Conv(x = graph_data.x, edge_index = graph_data.edge_index)
        pool_output = global_max_pool(x = gcn_output, batch = graph_data.batch)

        hidden = self.fc_network(pool_output)
        return hidden
    
    def get_value(self, unprocessed_obs):
        temp_values = []
        critic_values = []
        b_graph_obs = []
        
        batched_graph_obs = create_graph_input(unprocessed_obs)
        num_graphs = batched_graph_obs.num_graphs
        ptr = batched_graph_obs.ptr
        batch_outputs = self.forward(batched_graph_obs.to(device), num_graphs, ptr)
        
        critic_value = self.critic(batch_outputs.to(device))
        return critic_value

    def get_action_and_value(self, unprocessed_obs, action = None):
        temp_values = []
        actions = []
        critic_values = []
        b_graph_obs = []
        t_graph_obs = create_graph_input(unprocessed_obs)
        num_graphs = t_graph_obs.num_graphs
        ptr = t_graph_obs.ptr

        grap_conv_output = self.forward(t_graph_obs.to(device), num_graphs, ptr)
        logits = self.actor(grap_conv_output)
        critic_value = self.critic(grap_conv_output)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), critic_value


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    model_file = os.path.join(args.model_path, 'GATv2_Clean_FB_single_file_v1_10M')
    run_name = f"{args.env_id}__GATv2__FB__single_file_v1{args.exp_name}__{args.seed}__{int(time.time())}"
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
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

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
        
        # Save the model if required
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

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)

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
