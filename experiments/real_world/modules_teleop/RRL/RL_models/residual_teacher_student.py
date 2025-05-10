# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation

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
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-6) # TODO: remove /5
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

class ResidualStudentTeacher(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        student_hidden_size=512,
        student_num_layers=2,
        student_activation="ReLU",
        teacher_hidden_size=256,
        teacher_num_layers=2,
        teacher_activation="ReLU",
        init_noise_std=0.1,
        action_head_std=0.01, # initialization gain of last layer
        visual_size=120*120,
        encoder_output_dim=128,
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.loaded_teacher = False  # indicates if teacher has been loaded
        
        self.visual_size = visual_size
        self.visual_encoder = CNNEncoder(output_dim=encoder_output_dim)
        self.Height = 120
        self.Width = 120

        mlp_input_dim_s = num_student_obs - visual_size + encoder_output_dim     # vision policy
        mlp_input_dim_t = num_teacher_obs                                        # state-based policy

        # student
        self.student = build_mlp(
            input_dim=mlp_input_dim_s,
            hidden_sizes=[student_hidden_size] * student_num_layers,
            output_dim=num_actions,
            activation=student_activation,
            output_std=action_head_std,
            bias_on_last_layer=False,            
        )

        # teacher
        self.teacher = build_mlp(
            input_dim=mlp_input_dim_t,
            hidden_sizes=[teacher_hidden_size] * teacher_num_layers,
            output_dim=num_actions,
            activation=teacher_activation,
            output_std=action_head_std,
            bias_on_last_layer=False,            
        )
        self.teacher.eval()

        print(f"Student MLP: {self.student}")
        print(f"Teacher MLP: {self.teacher}")

        # action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None, hidden_states=None):
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
        # pass through encoder
        visual_obs = observations[:,-self.visual_size:].reshape(-1, 1, self.Height, self.Width)
        visual_embedding = self.visual_encoder(visual_obs)
        observations = torch.cat((observations[:,:-self.visual_size], visual_embedding), dim=-1)

        mean = self.student(observations)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        # pass through encoder
        visual_obs = observations[:,-self.visual_size:].reshape(-1, 1, self.Height, self.Width)
        visual_embedding = self.visual_encoder(visual_obs)
        observations = torch.cat((observations[:,:-self.visual_size], visual_embedding), dim=-1)
        
        actions_mean = self.student(observations)
        return actions_mean

    def evaluate(self, teacher_observations):
        with torch.no_grad():
            actions = self.teacher(teacher_observations)
        return actions

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the student and teacher networks.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters.
        """

        # check if state_dict contains teacher and student or just teacher parameters
        if any("actor" in key for key in state_dict.keys()):  # loading parameters from rl training
            # rename keys to match teacher and remove critic parameters
            teacher_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            # also load recurrent memory if teacher is recurrent
            if self.is_recurrent and self.teacher_recurrent:
                raise NotImplementedError("Loading recurrent memory for the teacher is not implemented yet")  # TODO
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            return False
        elif any("student" in key for key in state_dict.keys()):  # loading parameters from distillation training
            super().load_state_dict(state_dict, strict=strict)
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            return True
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass
