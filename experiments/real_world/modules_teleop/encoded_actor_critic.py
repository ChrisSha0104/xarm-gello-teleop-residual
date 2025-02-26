#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from .res_net import ResNet18Conv
from .res_net import CNNEncoder
import time

class ActorCriticVisual(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        init_noise_std=1.0,
        use_visual_encoder=True,
        visual_idx_actor=[56,56+120*120],
        visual_idx_critic=[64,64+120*120],
        encoder_output_dim=128,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        self.Height = 120
        self.Width = 120

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        self.use_visual_encoder = use_visual_encoder
        self.visual_idx_actor = visual_idx_actor
        self.visual_idx_critic = visual_idx_critic

        if use_visual_encoder:
            mlp_input_dim_a = mlp_input_dim_a - (visual_idx_actor[1] - visual_idx_actor[0]) + encoder_output_dim # 56+128 = 184
            mlp_input_dim_c = mlp_input_dim_c - (visual_idx_critic[1] - visual_idx_critic[0]) + encoder_output_dim # 56+8+128 = 192
            # ResNet feature extractor
            # self.visual_encoder = ResNet18Conv(
            #     input_channel=1,
            #     pretrained=True,
            #     mlp_input_dim=512*3*3, # mlp input for the MLP encoder, i.e., the output of the ResNet
            #     mlp_output_dim=encoder_output_dim  # Output dimension of the MLP encoder following ResNet
            # )
            self.visual_encoder = CNNEncoder(output_dim=encoder_output_dim) # input size (N,1,96,96)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        """
        Update the action distribution based on observations.
        """
        if self.use_visual_encoder:
            visual_obs = observations[:,self.visual_idx_actor[0]:self.visual_idx_actor[1]].reshape(-1, 1, self.Height, self.Width)
            visual_embedding = self.visual_encoder(visual_obs)
            visual_embedding = visual_embedding / (torch.norm(visual_embedding, dim=-1, keepdim=True)/5 + 1e-6)
            observations = torch.cat((observations[:,:self.visual_idx_actor[0]], visual_embedding), dim=-1)
        mean = self.actor(observations)  # Compute action mean
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        """
        Compute the mean action for inference (no sampling).
        """
        if self.use_visual_encoder:
            visual_obs = observations[:,self.visual_idx_actor[0]:self.visual_idx_actor[1]].reshape(-1, 1, self.Height, self.Width)
            visual_embedding = self.visual_encoder(visual_obs)
            visual_embedding = visual_embedding / (torch.norm(visual_embedding, dim=-1, keepdim=True)/5 + 1e-6)
            observations = torch.cat((observations[:,:self.visual_idx_actor[0]], visual_embedding), dim=-1)

        return self.actor(observations)
    
    def evaluate(self, critic_observations, **kwargs):
        """
        Evaluate the critic for the given observations.
        """
        if self.use_visual_encoder:
            visual_obs = critic_observations[:,self.visual_idx_critic[0]:self.visual_idx_critic[1]].reshape(-1, 1, self.Height, self.Width)
            visual_embedding = self.visual_encoder(visual_obs)
            visual_embedding = visual_embedding / (torch.norm(visual_embedding, dim=-1, keepdim=True)/5 + 1e-6)
            critic_observations = torch.cat((critic_observations[:,:self.visual_idx_critic[0]], visual_embedding), dim=-1)
        return self.critic(critic_observations)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
