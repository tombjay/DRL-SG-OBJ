import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

class SlotAttention(nn.Module):
    """Slot Attention module."""

    def __init__(self, num_iterations, num_slots, slot_size, mlp_hidden_size, epsilon=1e-8):
        """Builds the Slot Attention module.

        Args:
          num_iterations: Number of iterations.
          num_slots: Number of slots.
          slot_size: Dimensionality of slot feature vectors.
          mlp_hidden_size: Hidden layer size of MLP.
          epsilon: Offset for attention coefficients before normalization.
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(self.slot_size)
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_mlp = nn.LayerNorm(self.slot_size)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_k = nn.Linear(self.slot_size, self.slot_size, bias=False)
        self.project_v = nn.Linear(self.slot_size, self.slot_size, bias=False)

        # Parameters for Gaussian init (shared by all slots).
        self.slots_mu = nn.Parameter(torch.randn(1, 1, self.slot_size)).to(device)
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, self.slot_size)).to(device)
        
        # Slot update functions.
        self.gru = nn.GRU(self.slot_size, self.slot_size, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size)
        )

    def forward(self, inputs):
        # `inputs` has shape [batch_size, num_inputs, inputs_size].
        batch_size = inputs.shape[0]
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        k = self.project_k(inputs)  # Shape: [batch_size, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, num_inputs, slot_size].

        # Initialize the slots. Shape: [batch_size, num_slots, slot_size].
        slots = self.slots_mu + torch.exp(self.slots_log_sigma) * torch.randn(batch_size, self.num_slots, self.slot_size).to(device)
        initial_hidden_slot = self.slots_mu + torch.exp(self.slots_log_sigma) * torch.randn(1, batch_size, self.slot_size, device=device).to(device)

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            normalization = self.slot_size ** -0.5
            # q *= self.slot_size ** -0.5  # Normalization.
            q = q * normalization
            attn_logits = torch.bmm(k, q.transpose(1, 2))
            attn = F.softmax(attn_logits, dim=-1)
            # `attn` has shape: [batch_size, num_inputs, num_slots].

            # Weigted mean.
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            updates = torch.bmm(attn.transpose(1, 2), v)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            slots, _ = self.gru(updates,  initial_hidden_slot)
            slot_update = self.mlp(self.norm_mlp(slots)).to(device)
            # slots += self.mlp(self.norm_mlp(slots))
            slots = slots + slot_update

        return slots


def spatial_broadcast(slots, resolution):
    """Broadcast slot features to a 2D grid and collapse slot dimension."""
    # `slots` has shape: [batch_size, num_slots, slot_size].
    slots = slots.view(-1, slots.shape[-1])[:, None, None, :]
    grid = slots.expand(-1, resolution[0], resolution[1], -1)
    # `grid` has shape: [batch_size*num_slots, width, height, slot_size].
    return grid


def spatial_flatten(x):
    return x.reshape(-1, x.size(1) * x.size(2), x.size(-1))


def unstack_and_split(x, batch_size, num_channels=3):
    """Unstack batch dimension and split into channels and alpha mask."""
    unstacked = x.view(batch_size, -1, *x.shape[1:])
    channels, masks = torch.split(unstacked, [num_channels, 1], dim=-1)
    return channels, masks

def build_grid(resolution):
        ranges = [torch.linspace(0., 1., steps=res) for res in resolution]
        grid = torch.meshgrid(ranges)
        grid = torch.stack(grid, dim=-1)
        grid = grid.reshape(resolution[0], resolution[1], -1)
        grid = grid.unsqueeze(0)
        grid = torch.cat([grid, 1.0 - grid], dim=-1)
        return grid
    

class SlotAttentionClassifier(nn.Module):
    """Slot Attention-based classifier for property prediction."""

    def __init__(self, num_slots = 5, num_iterations = 3, slot_size = 64, mlp_hidden_size = 128):
        """Builds the Slot Attention-based classifier.

        Args:
          resolution: Tuple of integers specifying width and height of input image.
          num_slots: Number of slots in Slot Attention.
          num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        # self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU()
        )

        self.encoder_pos = SoftPositionEmbed(64, (32, 32))

        self.layer_norm = nn.LayerNorm(64, 64)
        self.mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        self.slot_attention = SlotAttention(
            num_iterations = num_iterations,
            num_slots = num_slots,
            slot_size = slot_size,
            mlp_hidden_size = mlp_hidden_size)


    def forward(self, frames):
        # `image` has shape: [batch_size, height, width, num_channels].
        # Convolutional encoder with position embedding.
        start_time = time.time()
        x = self.encoder_cnn(frames.permute(0,3,1,2))  # CNN Backbone.
        x = self.encoder_pos(x.permute(0, 3, 2, 1))  # Position embedding.
        x = spatial_flatten(x)  # Flatten spatial dimensions (treat image as set).
        x = self.mlp(self.layer_norm(x))  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        slots = self.slot_attention(x)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        return slots
    

class SoftPositionEmbed(nn.Module):
    """Adds soft positional embedding with learnable projection."""

    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.

        Args:
          hidden_size: Size of input feature dimension.
          resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.dense = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        return inputs + self.dense(self.grid.to(device))