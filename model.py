"""
Constructs a pytorch model for a convolutional neural network
    Usage: from model.target import target
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class CNNModel(nn.Module):
    def __init__(self):
        """
        Define the architecture, i.e. what layers our network contains.
        At the end of __init__() we call init_weights() to initialize all model parameters (weights and biases)
        in all layers to desired distributions.
        """
        super().__init__()

        self.conv1 = torch.nn.Conv1d(1, 8, 15, stride=1, padding=0)
        # 7986
        # 2662
        self.conv2 = torch.nn.Conv1d(8, 16, 11, stride=1, padding=0)
        # 2652
        # 884
        self.conv3 = torch.nn.Conv1d(16, 32, 9, stride=1, padding=0)
        # 876
        # 292
        self.conv4 = torch.nn.Conv1d(32, 64, 5, stride=1, padding=0)
        # 288
        # 96

        self.pool = torch.nn.MaxPool1d(3)

        self.fc_1 = torch.nn.Linear(6144, 256)
        self.fc_2 = torch.nn.Linear(256, 128)
        self.fc_3 = torch.nn.Linear(128, 10)

        self.dropout = torch.nn.Dropout(p=0.3)

        self.init_weights()

    def init_weights(self):
        """Initialize all model parameters (weights and biases) in all layers to desired distributions"""
        torch.manual_seed(42)

        for conv in [self.conv1, self.conv2, self.conv3, self.conv4]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / sqrt(5 * 5 * C_in))
            nn.init.constant_(conv.bias, 0.0)
        for fc in [self.fc_1, self.fc_2, self.fc_3]:
            nn.init.normal_(fc.weight, 0.0, 1 / sqrt(32))
            nn.init.constant_(fc.bias, 0.0)

    def forward(self, x):
        """
        This function defines the forward propagation for a batch of input examples, by
        successively passing output of the previous layer as the input into the next layer (after applying
        activation functions), and returning the final output as a torch.Tensor object.

        You may optionally use the x.shape variables below to resize/view the size of
        the input matrix at different points of the forward pass.
        """

        matrix = x
        relu = torch.nn.ReLU()
        soft = torch.nn.Softmax()

        for conv in [self.conv1, self.conv2, self.conv3, self.conv4]:
            matrix = relu(conv(matrix))
            matrix = self.pool(matrix)
            matrix = self.dropout(matrix)

        matrix = torch.flatten(matrix, 1)

        for fc in [self.fc_1, self.fc_2]:
            matrix = relu(fc(matrix))
            matrix = self.dropout(matrix)

        return soft(self.fc_3(matrix))
