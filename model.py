"""
Constructs a pytorch model for a convolutional neural network
    Usage: from model import CNNModel
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class CNNModel(nn.Module):
    def __init__(self):
        """
        Define the architecture, i.e. what layers our network contains.
        """
        super().__init__()

        # Convolutional layers
        self.conv1 = torch.nn.Conv1d(1, 8, 15, stride=1, padding=0)
        self.conv2 = torch.nn.Conv1d(8, 16, 11, stride=1, padding=0)
        self.conv3 = torch.nn.Conv1d(16, 32, 9, stride=1, padding=0)
        self.conv4 = torch.nn.Conv1d(32, 64, 5, stride=1, padding=0)

        # Pooling layer
        self.pool = torch.nn.MaxPool1d(3)

        # Fully connected layers
        self.fc_1 = torch.nn.Linear(6144, 256)
        self.fc_2 = torch.nn.Linear(256, 128)
        self.fc_3 = torch.nn.Linear(128, 10)

        # Dropout layers to reduce overfitting
        self.dropout1 = torch.nn.Dropout(p=0.1)
        self.dropout3 = torch.nn.Dropout(p=0.3)

    def forward(self, x):
        """
        Forward propagation for a batch of input examples, by passing output of the previous 
        layer as the input into the next layer (after applying activation functions), and 
        returning the final output as a torch.Tensor object.
        """

        matrix = x
        relu = torch.nn.ReLU()

        for conv in [self.conv1, self.conv2, self.conv3, self.conv4]:
            matrix = relu(conv(matrix))
            matrix = self.pool(matrix)
            matrix = self.dropout1(matrix)

        matrix = torch.flatten(matrix, 1)

        for fc in [self.fc_1, self.fc_2]:
            matrix = relu(fc(matrix))
            matrix = self.dropout3(matrix)

        return self.fc_3(matrix)
