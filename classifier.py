import torch
import torch.nn as nn
import torch.nn.functional as F


class ClfModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.mp1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv_drop = nn.Dropout2d()
        self.mp2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        
        self.fc1 = nn.Linear(320, 160)
        self.dp1 = nn.Dropout()
        self.fc2 = nn.Linear(160, 40)
        self.fc3 = nn.Linear(40, 4)
        self.apply(self.__init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.conv_drop(x)
        x = self.mp2(x)
        x = self.relu2(x)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dp1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def __init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
