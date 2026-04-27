# -*- coding: utf-8 -*-
"""
Created on Wed 01 Sept 2021

@author: Hakim Benkirane

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Sets-up the different types of layers that can be used.
"""

import torch.nn as nn
import torch.nn.init as init


class FullyConnectedLayer(nn.Module):
    """
    Linear => Norm1D => LeakyReLU
    """
    def __init__(self, input_dim, output_dim, norm_layer=nn.BatchNorm1d, leaky_slope=0.2, dropout=0.2, activation=True, normalization=True, activation_name='LeakyReLU'):
        """
        Construct a fully-connected block
        Parameters:
            input_dim (int)         -- the dimension of the input tensor
            output_dim (int)        -- the dimension of the output tensor
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            activation (bool)       -- need activation or not
            normalization (bool)    -- need normalization or not
            activation_name (str)   -- name of the activation function used in the FC block
        """
        super(FullyConnectedLayer, self).__init__()
        self.fc_block = []
        # Linear
        linear_layer = nn.Linear(input_dim, output_dim)
        
        init.kaiming_uniform_(linear_layer.weight, a=0.2, nonlinearity='leaky_relu')
        init.zeros_(linear_layer.bias)
        self.fc_block.append(linear_layer)

        # Norm
        if normalization:
            norm_layer = nn.BatchNorm1d
            self.fc_block.append(norm_layer(output_dim))
        # Dropout
        if 0 < dropout <= 1:
            self.fc_block.append(nn.Dropout(p=dropout))
        # LeakyReLU
        if activation:
            if activation_name.lower() == 'relu':
                self.fc_block.append(nn.ReLU())
            elif activation_name.lower() == 'sigmoid':
                self.fc_block.append(nn.Sigmoid())
            elif activation_name.lower() == 'leakyrelu':
                self.fc_block.append(nn.LeakyReLU(negative_slope=leaky_slope, inplace=False))
            elif activation_name.lower() == 'tanh':
                self.fc_block.append(nn.Tanh())
            elif activation_name.lower() == 'softmax':
                self.fc_block.append(nn.Softmax(dim=1))
            elif activation_name.lower() == 'no':
                pass
            else:
                raise NotImplementedError('Activation function [%s] is not implemented' % activation_name)

        self.fc_block = nn.Sequential(*self.fc_block)

    def forward(self, x):
        y = self.fc_block(x)
        return y

