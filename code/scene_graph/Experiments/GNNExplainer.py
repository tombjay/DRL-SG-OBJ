"""
This is an incomplete code file to test the Graph explainability.
The first step involves running the saved model using sample input frames from Atarihead dataset.
Note the actions taken and save the scene graph constructed as PyTorch Geometric's 'Batch'
Then pass the saved 'Batch' graph data into the GNNExplainer algorithm to generate the subgraphs.
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
from torch.distributions.categorical import Categorical
from torch.distributions.gumbel import Gumbel
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn.conv import GATv2Conv, GCNConv
from torch_geometric.nn import aggr, Sequential as Seq
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.nn.pool import global_mean_pool, global_max_pool
import matplotlib.pyplot as plt
import networkx as nx

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
    env_id: str = "ALE/SpaceInvaders-v5"
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
    num_envs: int = 1
    """the number of parallel game environments"""

    

global_color_embedding = [
    torch.tensor([ 73.9750,  32.5060, -25.0362]),
    torch.tensor([ 73.9750,  32.5060, -25.0362]),
    torch.tensor([ 92.6469,  36.9254, -12.9817]),
    torch.tensor([ 93.6313, 36.9862,  5.3785]),
    torch.tensor([ 92.5145, 42.5437, 12.3467]),
    torch.tensor([ 134.0074,  61.1213, -17.1469]),
    torch.tensor([ 109.6145,  45.0331,  -9.8320])
]

class NodeFeatureEmbedding(nn.Module):
    def __init__(self, bbox_in_dim = 4, bbox_out_dim = 4):
        super(NodeFeatureEmbedding, self).__init__()
        self.bbox_layer = nn.Linear(bbox_in_dim, bbox_out_dim)

    def forward(self, bounding_boxes):
        # Pass bounding boxes through linear layer
        bbox_embeddings = self.bbox_layer(bounding_boxes)
        return bbox_embeddings


def generate_bounding_boxes(frames):
    batch_boxes = []
    batch_labels = []
    frames = frames.cpu().numpy()
    # for frame in frames:
    boxes, labels = [], []

    for obj_class, color in object_colors.items():
        minx, miny, maxx, maxy, closing_dist = get_object_coordinates(obj_class)

        masks = [cv2.inRange(frames[miny:maxy, minx:maxx, :], np.array(color), np.array(color)) for color in color]
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
    
    bounding_box_embedding = NodeFeatureEmbedding()

    batch_boxes, batch_labels = generate_bounding_boxes(frames)
    print(batch_boxes, batch_labels)

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
            color_embed = global_color_embedding[label]

            # Encode whether the node is friendly or not
            if label in [1, 4, 6]:
                friendly_indicator = torch.tensor([1.])  # Friendly node
            else:
                friendly_indicator = torch.tensor([0.])  # Not friendly node

            # Concatenate bounding box coordinates, label encoding, and friendly indicator
            node_feature = torch.cat((node_feature, color_embed, torch.tensor([label])))

            node_features.append(node_feature)

        
        node_features = torch.stack(node_features)


        # Compute pairwise distances between all pairs of nodes
        num_nodes = node_features.size(0)

        # bounding boxes are represented by node_features[:, :4]
        bounding_boxes = torch.tensor(boxes)

        # Create mask for nodes with label 1
        labels_tensor = torch.tensor(labels)
        label_1_mask = labels_tensor == 0

        # Create adjacency matrix
        adjacency_matrix = torch.zeros(num_nodes, num_nodes, dtype=torch.int)

        # Connect nodes with label 1 to all other nodes
        adjacency_matrix[label_1_mask, :] = 1

        adjacency_matrix.view(-1)[::num_nodes + 1] = 0
        
        # Convert adjacency matrix to edge index format
        edge_index = adjacency_matrix.nonzero(as_tuple=False).t()

        batch_node_features.append(node_features)
        batch_edge_index.append(edge_index)

    batch_data_list = [Data(x=node_features, edge_index=edge_index) for node_features, edge_index in zip(batch_node_features, batch_edge_index)]
    batch = Batch.from_data_list(batch_data_list)
    # torch.save(batch, '/scratch/users/sundararaj/msc2023_jayakumar/fasterRCNN/sgg/batched_data/batch_data12.pt')
    
    return batch

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
    "player": 0,
    "score": 1, 
    "alien": 2, 
    "shield": 3, 
    "satellite": 4, 
    "bullet": 5, 
    "lives": 6
}

# Remove unwanted objects in the frame.
labels_to_remove = [1, 6]

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

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, input_node_dim = 8, hidden_dim = 128, gcn_layers = 2):
        super().__init__()
        
        self.GCNConv = Seq('x, edge_index', [
            (GCNConv(in_channels = input_node_dim, out_channels = hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
            (GCNConv(in_channels = hidden_dim, out_channels = hidden_dim), 'x, edge_index -> x'),
            nn.ReLU(inplace=True),
        ])
        
        self.fc_network = nn.Sequential(
            # nn.Flatten(),
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

    def forward(self, x, edge_index, graph_data):
        gcn_output = self.GCNConv(x = x, edge_index = edge_index)
        pool_output = global_max_pool(x = gcn_output, batch = graph_data.batch)

        hidden = self.fc_network(pool_output)
        return hidden
    
    def get_value(self, unprocessed_obs):
        
        batched_graph_obs = create_graph_input(unprocessed_obs).to(device)
        batch_outputs = self.forward(batched_graph_obs.x, batched_graph_obs.edge_index, batched_graph_obs)
        
        critic_value = self.critic(batch_outputs.to(device))
        return critic_value

    def get_action_and_value(self, unprocessed_obs, action = None):
        t_graph_obs = create_graph_input(unprocessed_obs).to(device)

        grap_conv_output = self.forward(t_graph_obs.x, t_graph_obs.edge_index, t_graph_obs)
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

def plot_original(graph_data):

    # Your PyTorch Geometric Data object
    data = graph_data

    # Convert the edge index to an edge list
    edge_list = data.edge_index.t().tolist()

    # Create a graph from the edge list
    G = nx.Graph(edge_list)

    # Extract node features
    node_features = data.x.numpy()

    # Map node types to colors and labels
    type_to_color = {
        0: 'yellow',    # player
        1: 'green',  # score
        2: 'blue',   # alien
        3: 'purple', # shield
        4: 'orange', # satellite
        5: 'pink',   # bullet
        6: 'cyan'   # lives
    }

    type_to_label = {
        0: 'Player',
        1: 'Score',
        2: 'Alien',
        3: 'Shield',
        4: 'Satellite',
        5: 'Bullet',
        6: 'Lives'
    }

    # Assign colors to nodes based on their types
    node_colors = [type_to_color[int(node[-1])] for node in node_features]

    # Draw the graph with different node colors
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=500, node_color=node_colors, font_color='black')

    # Create a legend for node types and colors
    legend_handles = []
    for node_type, color in type_to_color.items():
        legend_handles.append(plt.Line2D([0], [0], marker='o', color=color, label=type_to_label[node_type], markersize=10))

    # Display the legend inside the graph
    plt.legend(handles=legend_handles, title='Node Types', loc='upper left', bbox_to_anchor=(1, 1))

    # Display the plot
    plt.savefig('original_plot4.pdf')

if __name__ == "__main__":
    args = tyro.cli(Args)
    model_file = os.path.join(args.model_path, 'GCN_SPI_v8_star_10M')
    run_name = f"{args.env_id}_GCN_SPI_v8_star_10M_{args.exp_name}__{args.seed}__{int(time.time())}"
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
    
    
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"


    agent = Agent(envs).to(device)
    
    print("Loading model from:", model_file)
    agent.load_state_dict(torch.load(model_file))
    
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
    
    saved_actions = []
    image_path = r'/scratch/users/sundararaj/msc2023_jayakumar/fasterRCNN/sgg/Saliency_image/SI_12.png'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (160, 210), cv2.INTER_AREA)
    obs = torch.Tensor(image_resized).to(device)
    
    
    # Uncomment this code to generate actions for input samples
    # with torch.no_grad():
    #     action, _, _, _ = agent.get_action_and_value(obs)
    #     print('action',action)
    
    dataset = torch.load('/scratch/users/sundararaj/msc2023_jayakumar/fasterRCNN/sgg/batched_data/batch_data12.pt')
    # plot_original(data)
    
    explainer = Explainer(
    model=agent,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type="attributes",
    edge_mask_type="object",
    model_config=dict(
        mode='multiclass_classification',
        task_level='graph',
        return_type='probs',
    ),
    )
    explanation = explainer(x=dataset.x.to(device), edge_index = dataset.edge_index.to(device), graph_data = dataset.to(device))
    print("--------------------------")
    print('explanation', explanation.node_mask)
    print("--------------------------")
    
    print(f'Generated explanations in {explanation.available_explanations}')
    
    
    path = 'subgraph12.pdf'
    explanation.visualize_graph(path, backend = "networkx")
    print(f"Subgraph visualization plot has been saved to '{path}'")