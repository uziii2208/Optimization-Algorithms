import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 1 input channel (grayscale), 32 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, 3)  # 28x28 -> 26x26
        self.conv2 = nn.Conv2d(32, 64, 3)  # 13x13 -> 11x11
        # Calculate size after convolutions and pooling
        # After conv1: 26x26
        # After maxpool: 13x13
        # After conv2: 11x11
        # After maxpool: 5x5
        # Therefore final size is 64 channels * 5 * 5
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 26x26 -> 13x13
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 11x11 -> 5x5
        x = x.view(-1, 64 * 5 * 5)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x