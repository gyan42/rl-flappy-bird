import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):

    def __init__(self,
                 number_of_actions,
                 gamma,
                 initial_epsilon,
                 final_epsilon,
                 replay_memory_size,
                 batch_size):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = number_of_actions
        self.gamma = gamma
        self.final_epsilon = final_epsilon
        self.initial_epsilon = initial_epsilon
        self.replay_memory_size = replay_memory_size
        self.minibatch_size = batch_size

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)

        return out