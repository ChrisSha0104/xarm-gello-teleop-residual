#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
import time
from typing import Tuple

'''
Uses rsl-rl actor critic with visual encoder and residual policy related designs (e.g., orthogonal initialization, small gain factors, low initial standard deviation, and smooth activations (SiLU))
'''

class CNNEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super(CNNEncoder, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),  # (32, 23, 23)
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # (64, 12, 12)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # (64, 12, 12)
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.AdaptiveAvgPool2d((6, 6))  # Downsample to (64, 6, 6)
        )
        
        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 256),  # Fully connected layer
            nn.ReLU(),
            nn.Linear(256, output_dim)   # Map to 128 dimensions
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        x = x / (x.norm(dim=-1, keepdim=True)/5 + 1e-6)
        return x

def layer_init(layer, nonlinearity="ReLU", std=np.sqrt(2), bias_const=0.0):
    '''
    centralize weight/bias initialization for all linear layers
    '''
    if isinstance(layer, nn.Linear):
        if nonlinearity == "ReLU":
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
        elif nonlinearity == "SiLU":
            nn.init.kaiming_normal_(
                layer.weight, mode="fan_in", nonlinearity="relu"
            )  # Use relu for Swish
        elif nonlinearity == "Tanh":
            torch.nn.init.orthogonal_(layer.weight, std)
        else:
            nn.init.xavier_normal_(layer.weight)

    # Only initialize the bias if it exists
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)

    return layer


def build_mlp(
    input_dim,
    hidden_sizes,
    output_dim,
    activation,
    output_std=1.0,
    bias_on_last_layer=True,
    last_layer_bias_const=0.0,
):
    act_func = getattr(nn, activation)
    layers = []
    layers.append(
        layer_init(nn.Linear(input_dim, hidden_sizes[0]), nonlinearity=activation)
    )
    layers.append(act_func())
    for i in range(1, len(hidden_sizes)):
        layers.append(
            layer_init(
                nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]), nonlinearity=activation
            )
        )
        layers.append(act_func())
        
    # orthogonal init with small initial std
    layers.append( 
        layer_init(
            nn.Linear(hidden_sizes[-1], output_dim, bias=bias_on_last_layer),
            std=output_std,
            nonlinearity="Tanh",
            bias_const=last_layer_bias_const,
        )
    )
    return nn.Sequential(*layers)

class ResidualActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_size=256,
        actor_num_layers=2,
        actor_activation="SiLU",
        critic_hidden_size=256,
        critic_num_layers=2,
        critic_activation="SiLU",
        init_logstd=-3,
        action_head_std=0.01, # initialization gain of last layer
        action_scale=0.1, # scale residual in env
        critic_last_layer_bias_const=0.0,
        critic_last_layer_std=1.0,
        critic_last_layer_activation=None,
        use_visual_encoder=True,
        visual_idx_actor=[60,60+120*120],
        visual_idx_critic=[73,73+120*120],
        encoder_output_dim=128,
        learn_std=True,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        if num_actor_obs == num_critic_obs:
            print("using symmetric actor critic")
        else:
            print("using asymmetric actor critic")

        self.Height = 120
        self.Width = 120

        self.use_visual_encoder = use_visual_encoder
        self.visual_idx_actor = visual_idx_actor
        self.visual_idx_critic = visual_idx_critic

        if self.use_visual_encoder:
            mlp_input_dim_a = num_actor_obs - (visual_idx_actor[1] - visual_idx_actor[0]) + encoder_output_dim # 56+128 = 184
            mlp_input_dim_c = num_critic_obs - (visual_idx_critic[1] - visual_idx_critic[0]) + encoder_output_dim # 56+8+128 = 192
            self.visual_encoder = CNNEncoder(output_dim=encoder_output_dim) # input size (N,1,96,96)
        else:
            mlp_input_dim_a = num_actor_obs
            mlp_input_dim_c = num_critic_obs
            
        # Policy
        self.actor = build_mlp(
            input_dim=mlp_input_dim_a,
            hidden_sizes=[actor_hidden_size] * actor_num_layers,
            output_dim=num_actions,
            activation=actor_activation,
            output_std=action_head_std,
            bias_on_last_layer=False,            
        )

        self.critic = build_mlp(
            input_dim=mlp_input_dim_c,
            hidden_sizes=[critic_hidden_size] * critic_num_layers,
            output_dim=1,
            activation=critic_activation,
            output_std=critic_last_layer_std,
            bias_on_last_layer=True,
            last_layer_bias_const=critic_last_layer_bias_const,
        )

        if critic_last_layer_activation is not None:
            self.critic.add_module(
                "output_activation",
                getattr(nn, critic_last_layer_activation)(),
            )

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # learn std
        self.actor_logstd = nn.Parameter(
            torch.ones(1, num_actions) * init_logstd,
            requires_grad=learn_std, # TODO: CHECKTHIS!!!!!!!!!!!!!!
        )
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False # type: ignore


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
    
    # ------------------------------------------------------------------

    def update_distribution(self, observations):
        """
        Update the action distribution based on observations.
        """
        if self.use_visual_encoder:
            visual_obs = observations[:,self.visual_idx_actor[0]:self.visual_idx_actor[1]].reshape(-1, 1, self.Height, self.Width)
            visual_embedding = self.visual_encoder(visual_obs)
            observations = torch.cat((observations[:,:self.visual_idx_actor[0]], visual_embedding), dim=-1)
        mean = self.actor(observations)  # Compute action mean
        log_std = self.actor_logstd.expand_as(mean)  # Learnable log standard deviation
        std = torch.exp(log_std)  # Convert log standard deviation to positive scale
        self.distribution = Normal(mean, std)

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
            observations = torch.cat((observations[:,:self.visual_idx_actor[0]], visual_embedding), dim=-1)
        return self.actor(observations)
    
    def evaluate(self, critic_observations, **kwargs):
        """
        Evaluate the critic for the given observations.
        """
        if self.use_visual_encoder:
            visual_obs = critic_observations[:,self.visual_idx_critic[0]:self.visual_idx_critic[1]].reshape(-1, 1, self.Height, self.Width)
            visual_embedding = self.visual_encoder(visual_obs)
            critic_observations = torch.cat((critic_observations[:,:self.visual_idx_critic[0]], visual_embedding), dim=-1)
        return self.critic(critic_observations)
