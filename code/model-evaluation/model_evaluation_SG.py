"""
--> This file consists the code for evaluating scene-graph models.
--> Change the path of saved model in 'model_file' variable and change the game environment in 'env_id' variable.
--> Copy & paste the suitable agent architecture from their respective files to evaluate. 
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
from torch.distributions.gumbel import Gumbel
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

class NodeFeatureEmbedding(nn.Module):
    def __init__(self, bbox_in_dim = 4, bbox_out_dim = 4):
        super(NodeFeatureEmbedding, self).__init__()
        self.bbox_layer = nn.Linear(bbox_in_dim, bbox_out_dim)

    def forward(self, bounding_boxes):
        # Pass bounding boxes through linear layer
        bbox_embeddings = self.bbox_layer(bounding_boxes)
        return bbox_embeddings

def assert_in(observed, expected, tol):
    """
    Asserts if the observed point is equal to the expected one with a given tolerance.
    Used for detecting certain objects in a frame
    True if ||observed - expected|| <= tol, with || the maximum over the two dimensions.

    :param observed: The observed value point (e.g. (x,y), (w,h))
    :type observed: (int, int)
    :param expected: The expected value point (also (x,y), (w,h))
    :type expected: (int, int)
    :param tol: A given tolerance.
    :type tol: int or (int, int)
    
    :return: True if points within the tolerance
    :rtype: bool
    """
    if type(tol) is int:
        tol = (tol, tol)
    return np.all([expected[i] + tol[i] >= observed[i] >= expected[i] - tol[i] for i in range(2)])

# Encoded color values
global_color_embedding = [
    torch.tensor([ 136.8210,  94.3227, 156.2151]),
    torch.tensor([ 66.2577, 56.3960, 60.7914]),
    torch.tensor([ 114.3690,  50.1826, 121.6223]),
    torch.tensor([ 53.5577, 47.4976, 67.4303]),
    torch.tensor([ 113.6652,  55.4476,  98.9007]),
    torch.tensor([ 40.1900, -57.3022,  -3.0790]),
    torch.tensor([ 75.5766, 60.9537, 81.9096]),
    torch.tensor([ 78.1941, -45.2639,  32.9210]),
    torch.tensor([ 79.5652, 56.0424, 71.4776]),
    torch.tensor([ 86.4535, 40.4111, 94.8505]),
    torch.tensor([ 112.9831,  81.0633, 130.8815]),
    torch.tensor([ 113.6652,  55.4476,  98.9007]),
    torch.tensor([ 113.6652,  55.4476,  98.9007])
]

def generate_bounding_boxes(frames):
    batch_boxes = []
    batch_labels = []
    frames = frames.cpu().numpy()
    for frame in frames:
        boxes, labels = [], []

        for obj_class, color in object_colors.items():
            minx, miny, maxx, maxy, closing_dist, size = get_object_coordinates(obj_class)

            masks = [cv2.inRange(frame[miny:maxy, minx:maxx, :], np.array(color), np.array(color)) for color in color]
            mask = sum(masks)

            closed = closing(mask, square(closing_dist))

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                x, y = x + minx, y + miny
                if size:
                    if not assert_in((w, h), size, 4):
                        continue
                boxes.append([x, y, x + w, y + h])
                labels.append(object_id_mapping[obj_class])

        filtered_boxes = [box for box, label in zip(boxes, labels) if label not in labels_to_remove]
        filtered_labels = [label for label in labels if label not in labels_to_remove]

        
        batch_boxes.append(filtered_boxes)
        batch_labels.append(filtered_labels)

    return batch_boxes, batch_labels

def create_graph_input(frames):
    
    bounding_box_embedding = NodeFeatureEmbedding()

    batch_boxes, batch_labels = generate_bounding_boxes(frames)

    batch_node_features = []
    batch_edge_index = []

    for boxes, labels in zip(batch_boxes, batch_labels):
        node_features = []
        
        for i, label in enumerate(labels):
            # node_feature = torch.tensor(boxes[i])
            bbox = torch.tensor(boxes[i], dtype=torch.float32)

            # Pass bounding box through linear layer to get embeddings
            node_feature = bounding_box_embedding.forward(bbox)
            
            # Get global color embedding
            color_embed = global_color_embedding[label - 1]

            # Encode whether the node is friendly or not
            if label == 1:
                friendly_indicator = torch.tensor([1.])  # Friendly node
            else:
                friendly_indicator = torch.tensor([0.])  # Not friendly node

            # Concatenate bounding box coordinates, label encoding, and friendly indicator
            node_feature = torch.cat((node_feature, color_embed, friendly_indicator))

            node_features.append(node_feature)

        
        node_features = torch.stack(node_features)


        # Compute pairwise distances between all pairs of nodes
        num_nodes = node_features.size(0)

        # bounding boxes are represented by node_features[:, :4]
        bounding_boxes = torch.tensor(boxes)

        # Compute centroids of bounding boxes
        # centroids = (bounding_boxes[:, :2] + bounding_boxes[:, 2:]) / 2  # Shape: (num_nodes, 2)

        # Compute pairwise differences between centroids
        # diffs = centroids.unsqueeze(1) - centroids.unsqueeze(0)  # Shape: (num_nodes, num_nodes, 2)

        # Compute pairwise distances along the last dimension (Euclidean distance)
        # pairwise_distances = torch.norm(diffs, dim=-1)  # Shape: (num_nodes, num_nodes)

        # Set distance threshold
        # distance_threshold = 30

        # Create mask for nodes with label 1
        labels_tensor = torch.tensor(labels)
        label_1_mask = labels_tensor == 1

        # Create mask for nodes based on distance threshold
        # distance_mask = pairwise_distances < distance_threshold

        # Create adjacency matrix
        adjacency_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.int)

        # Connect nodes with label 1 to all other nodes
        adjacency_matrix[label_1_mask, :] = 1

        # Connect nodes based on distance threshold
        # adjacency_matrix[~label_1_mask & distance_mask] = 1

        adjacency_matrix.view(-1)[::num_nodes + 1] = 0
        
        # Convert adjacency matrix to edge index format
        edge_index = adjacency_matrix.nonzero(as_tuple=False).t()

        batch_node_features.append(node_features)
        batch_edge_index.append(edge_index)

    batch_data_list = [Data(x=node_features, edge_index=edge_index) for node_features, edge_index in zip(batch_node_features, batch_edge_index)]
    batch = Batch.from_data_list(batch_data_list)

    return batch

# Define object colors and mappings
object_colors = {
    "car1": [[167, 26, 26]], 
    "car2": [[180, 231, 117]],
    "car3": [[105, 105, 15]],
    "car4": [[228, 111, 111]],
    "car5": [[24, 26, 167]],
    "car6": [[162, 98, 33]],
    "car7": [[84, 92, 214]], 
    "car8": [[184, 50, 50]], 
    "car9": [[135, 183, 84]],
    "car10": [[210, 210, 64]],
    "score": [[228, 111, 111]],
    "logo": [[228, 111, 111]],
    "chicken": [[252, 252, 84]]
}

object_id_mapping = {
    "background": 0,
    "chicken": 1,
    "car1": 2,
    "car2": 3,
    "car3": 4,
    "car4": 5,
    "car5": 6,
    "car6": 7,
    "car7": 8,
    "car8": 9,
    "car9": 10,
    "car10": 11,
    "score": 12,
    "logo": 13
}

# Remove unwanted objects in the frame.
labels_to_remove = [12, 13]

def get_object_coordinates(obj_class):
    # Default coordinates
    minx, miny, maxx, maxy, closing_dist, size = 0, 0, 160, 210, 3, None

    # Adjust coordinates based on object class
    if obj_class[:3] == "car":
        miny, maxy= 20, 184
    elif obj_class == "score":
        maxy= 14
    elif obj_class == "chicken":
        size = (7,8)
    
    return minx, miny, maxx, maxy, closing_dist, size

def closing(mask, kernel):
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def square(size):
    return np.ones((size, size), dtype=np.uint8)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, input_node_dim = 8, hidden_dim = 128):
        super().__init__()
        
        self.GCNConv = Seq('x, edge_index', [
            (GCNConv(in_channels = input_node_dim, out_channels = hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GCNConv(in_channels = hidden_dim, out_channels = hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
        ])
        
        self.fc_network = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 32)),
            nn.ReLU(),
        )
        
        self.actor = layer_init(nn.Linear(32, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(32, 1), std=1)

    def forward(self, graph_data, num_graphs, ptr):
        
        gcn_output = self.GCNConv(x = graph_data.x, edge_index = graph_data.edge_index)
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
    model_file = os.path.join(args.model_path, 'GCN_Clean_FW_star_topology_file_v2_10M')
    run_name = f"{args.env_id}_evaluate_baseline_FB{args.exp_name}__{args.seed}__{int(time.time())}"
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


    agent = Agent(envs).to(device)
    
    print("Loading model from:", model_file)
    agent.load_state_dict(torch.load(model_file))
    agent.eval()
    
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
    
    mean_reward, std_dev = evaluate_model(envs, agent)
    print('Evaluation rewards', mean_reward, 'Std.deviation', std_dev)