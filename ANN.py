import torch
import torch.nn as nn

class Simple_NN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.layers = nn.Sequential(
            nn.Linear(input_dim[0], 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU()
            nn.Linear(64, self.num_actions)
        )

    def forward(self, x):
        return self.layers(x)

class CNN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

        def feature_size(self):
            return self.conv_layers(torch.zeros(1, *self.input_dim)).view(1,-1).size(1)

        def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
            x = self.fc_layers(x)
            return x

class Dueling_Simple_NN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.feature_layers = nn.Sequential(
            nn.Linear(input_dim[0], 64),
            nn.ReLU(),
        )
        self.advantage_layers = nn.Sequential(
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )
        self.value_layers = nn.Sequential(
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,1),
        )

    def forward(self, x):
        x = self.feature_layers(x)
        advantage = self.advantage_layers(x)
        value = self.value_layers(x)
        return value + advantage - advantage.mean()

