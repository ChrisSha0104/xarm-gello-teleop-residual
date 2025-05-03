import math
import abc
import numpy as np
import textwrap
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models as vision_models

class Module(torch.nn.Module):
    """
    Base class for networks. The only difference from torch.nn.Module is that it
    requires implementing @output_shape.
    """
    @abc.abstractmethod
    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError

class ConvBase(Module):
    """
    Base class for ConvNets.
    """
    def __init__(self):
        super(ConvBase, self).__init__()

    # dirty hack - re-implement to pass the buck onto subclasses from ABC parent
    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        raise NotImplementedError

    def forward(self, inputs):
        x = self.nets(inputs)
        if list(self.output_shape(list(inputs.shape)[1:])) != list(x.shape)[1:]:
            raise ValueError('Size mismatch: expect size %s, but got size %s' % (
                str(self.output_shape(list(inputs.shape)[1:])), str(list(x.shape)[1:]))
            )
        return x

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
        return x

class ResNetDepthEncoder(nn.Module):
    def __init__(self, 
                 output_dim=128, 
                 pretrained=True):
        super().__init__()
        resnet = vision_models.resnet18(pretrained=pretrained)
        resnet.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(),
            nn.Linear(512, output_dim),
        )

    def forward(self, depth_img):
        # depth_img: (B, 120*120) with values in [0,1]
        B = depth_img.shape[0]
        x = depth_img.view(B, 1, 120, 120)   # ← reshape into (B,1,H,W)
        return self.encoder(x)
    



class ResNet18Conv(ConvBase):
    """
    A ResNet18 block that can be used to process input images.
    """
    def __init__(
        self,
        input_channel=3,
        pretrained=False,
        input_coord_conv=False,
        mlp_input_dim=512*3*3,
        mlp_output_dim=128,
    ):
        """
        Args:
            input_channel (int): number of input channels for input images to the network.
                If not equal to 3, modifies first conv layer in ResNet to handle the number
                of input channels.
            pretrained (bool): if True, load pretrained weights for all ResNet layers.
            input_coord_conv (bool): if True, use a coordinate convolution for the first layer
                (a convolution where input channels are modified to encode spatial pixel location)
        """
        super(ResNet18Conv, self).__init__()
        net = vision_models.resnet18(pretrained=True)

        if input_coord_conv:
            net.conv1 = CoordConv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        elif input_channel != 3:
            net.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # cut the last fc layer
        self._input_coord_conv = input_coord_conv
        self._input_channel = input_channel
        self.nets = torch.nn.Sequential(*(list(net.children())[:-2]))
        self.mlp_output_dim = mlp_output_dim

        # # Define an MLP to reduce dimensions
        self.mlp = nn.Sequential(
            nn.Flatten(),  # Flatten the spatial dimensions (B, C, H, W) → (B, C*H*W), i.e., (num_envs, 512*3*3)
            nn.Linear(mlp_input_dim, 1024),
            nn.ReLU(),  # Activation
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512, mlp_output_dim)  # Reduce to desired output dimension
        )

    def forward(self, x): 
        """
        Forward pass through the encoder and the MLP.
        """
        features = self.nets(x)
        reduced_features = self.mlp(features)
        return reduced_features


    # def output_shape(self, input_shape):
    #     """
    #     Function to compute output shape from inputs to this module. 

    #     Args:
    #         input_shape (iterable of int): shape of input. Does not include batch dimension.
    #             Some modules may not need this argument, if their output does not depend 
    #             on the size of the input, or if they assume fixed size input.

    #     Returns:
    #         out_shape ([int]): list of integers corresponding to output shape
    #     """
    #     assert(len(input_shape) == 3)
    #     out_h = int(math.ceil(input_shape[1] / 32.))
    #     out_w = int(math.ceil(input_shape[2] / 32.))
    #     return [512, out_h, out_w]

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        return header + '(input_channel={}, input_coord_conv={})'.format(self._input_channel, self._input_coord_conv)
