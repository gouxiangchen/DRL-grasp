import torch.nn as nn
from torch.distributions import Normal
import torch
import numpy as np


class MLPPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MLPPolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 100)
        self.relu1 = nn.ReLU()
        self.fc_mean = nn.Linear(100, action_dim)
        self.tanh = nn.Tanh()
        self.fc_std = nn.Linear(100, action_dim)
        self.sigmoid = nn.Softplus()

        nn.init.kaiming_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc_mean.weight.data)
        nn.init.xavier_normal_(self.fc_std.weight.data)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        mean = self.tanh(self.fc_mean(x))
        std = self.sigmoid(self.fc_std(x)) + 1e-5
        return mean, std

    def choose_action(self, state):
        mean, std = self.forward(state)
        dis = Normal(mean, std)
        return dis.sample().numpy()


class MLPValue(nn.Module):
    def __init__(self, state_dim):
        super(MLPValue, self).__init__()
        self.fc1 = nn.Linear(state_dim, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 1)
        self.tanh = nn.Tanh()

        nn.init.kaiming_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MLPRotation(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(MLPRotation, self).__init__()
        self.fc1 = nn.Linear(state_dim, 100)
        self.relu1 = nn.ReLU()
        self.fc_mean = nn.Linear(100, action_dim)
        self.tanh = nn.Tanh()
        self.fc_std = nn.Linear(100, action_dim)
        self.sigmoid = nn.Softplus()

        nn.init.kaiming_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc_mean.weight.data)
        nn.init.xavier_normal_(self.fc_std.weight.data)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        mean = self.tanh(self.fc_mean(x))
        std = self.sigmoid(self.fc_std(x)) + 1e-5
        return mean, std

    def choose_action(self, state):
        mean, std = self.forward(state)
        dis = Normal(mean, std)
        return dis.sample().numpy()


class MLPRotationValue(nn.Module):
    def __init__(self, state_dim):
        super(MLPRotationValue, self).__init__()
        self.fc1 = nn.Linear(state_dim, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 1)
        self.tanh = nn.Tanh()

        nn.init.kaiming_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CNNPolicy(nn.Module):
    def __init__(self, action_dim):
        super(CNNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=6, stride=2)  # 224 * 224 * 1 -> 110 * 110 * 32
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)  # 110 * 110 * 32 -> 55 * 55 * 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 55 * 55 * 32 -> 55 * 55 * 64
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2)  # 55 * 55 * 64 -> 26 * 26 * 64
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(26 * 26 * 64, 1024)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 64)
        self.relu5 = nn.ReLU()
        self.fc_mean = nn.Linear(64 + 2, action_dim)  # +3 for 3d position input
        self.tanh = nn.Tanh()
        self.fc_std = nn.Linear(64 + 2, action_dim)
        self.sigmoid = nn.Softplus()

        nn.init.kaiming_normal_(self.conv1.weight.data)
        nn.init.kaiming_normal_(self.conv2.weight.data)
        nn.init.kaiming_normal_(self.conv3.weight.data)
        nn.init.kaiming_normal_(self.fc1.weight.data)
        nn.init.kaiming_normal_(self.fc2.weight.data)
        nn.init.xavier_normal_(self.fc_mean.weight.data)
        nn.init.xavier_normal_(self.fc_std.weight.data)

    def forward(self, x, position):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.fc1(x.view(x.size(0), -1))
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.relu5(x)
        x = torch.cat((x, position), 1)
        mean = self.tanh(self.fc_mean(x))
        std = self.sigmoid(self.fc_std(x)) + 1e-5
        return mean, std

    def choose_action(self, state):
        mean, std = self.forward(state)
        dis = Normal(mean, std)
        return dis.sample().numpy()


class CNNRotation(nn.Module):
    def __init__(self, action_dim):
        super(CNNRotation, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=6, stride=2)  # 224 * 224 * 1 -> 110 * 110 * 32
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)  # 110 * 110 * 32 -> 55 * 55 * 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 55 * 55 * 32 -> 55 * 55 * 64
        self.relu2 = nn.ReLU()

        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # 55 * 55 * 64 -> 55 * 55 * 64
        self.relu2_2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2)  # 55 * 55 * 64 -> 26 * 26 * 64
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(26 * 26 * 64, 1024)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 64)
        self.relu5 = nn.ReLU()
        self.fc_mean = nn.Linear(64 + 1, action_dim)  # +3 for 3d position input
        self.tanh = nn.Tanh()
        self.fc_std = nn.Linear(64 + 1, action_dim)
        self.sigmoid = nn.Softplus()

        nn.init.xavier_normal_(self.conv1.weight.data)
        nn.init.xavier_normal_(self.conv2.weight.data)
        nn.init.xavier_normal_(self.conv2_2.weight.data)
        nn.init.xavier_normal_(self.conv3.weight.data)
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.xavier_normal_(self.fc_mean.weight.data)
        nn.init.xavier_normal_(self.fc_std.weight.data)

    def forward(self, x, rotation):
        # torch.set_printoptions(threshold=np.inf)
        x = self.conv1(x)
        # x.register_hook(lambda g: print('model x1 before gradient: ', g))
        x = self.relu1(x)
        # x.register_hook(lambda g: print('model x1 gradient: ', g))
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        # x.register_hook(lambda g: print('model x2 gradient: ', g))
        X = self.conv2_2(x)
        x = self.relu2_2(x)
        # x.register_hook(lambda g: print('model x3 gradient: ', g))
        x = self.conv3(x)
        x = self.relu3(x)
        # x.register_hook(lambda g: print('model x4 gradient: ', g))
        x = self.fc1(x.view(x.size(0), -1))
        x = self.relu4(x)
        # x.register_hook(lambda g: print('model x5 gradient: ', g))
        x = self.fc2(x)
        x = self.relu5(x)
        # x.register_hook(lambda g: print('model x6 gradient: ', g))
        x = torch.cat((x, rotation), 1)
        mean = self.tanh(self.fc_mean(x))
        # mean.register_hook(lambda g: print('model mean gradient: ', g))
        std = self.sigmoid(self.fc_std(x)) + 1e-5
        # std.register_hook(lambda g: print('model std gradient: ', g))
        return mean, std

    def choose_action(self, state):
        mean, std = self.forward(state)
        dis = Normal(mean, std)
        return dis.sample().numpy()


class CNNRotationBig(nn.Module):
    def __init__(self, action_dim):
        super(CNNRotationBig, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=65, stride=1)  # 224 * 224 * 1 -> 160 * 160 * 32
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)  # 160 * 160 * 32 -> 80 * 80 * 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=21, stride=1)  # 80 * 80 * 32 -> 60 * 60 * 64
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=6, stride=2)  # 60 * 60 * 64 -> 28 * 28 * 64
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(28 * 28 * 64, 1024)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 64)
        self.relu5 = nn.ReLU()
        self.fc_mean = nn.Linear(64 + 1, action_dim)  # +3 for 3d position input
        self.tanh = nn.Tanh()
        self.fc_std = nn.Linear(64 + 1, action_dim)
        self.sigmoid = nn.Softplus()

        nn.init.kaiming_normal_(self.conv1.weight.data)
        nn.init.kaiming_normal_(self.conv2.weight.data)
        nn.init.kaiming_normal_(self.conv3.weight.data)
        nn.init.kaiming_normal_(self.fc1.weight.data)
        nn.init.kaiming_normal_(self.fc2.weight.data)
        nn.init.xavier_normal_(self.fc_mean.weight.data)
        nn.init.xavier_normal_(self.fc_std.weight.data)

    def forward(self, x, rotation):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.fc1(x.view(x.size(0), -1))
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.relu5(x)
        x = torch.cat((x, rotation), 1)
        mean = self.tanh(self.fc_mean(x))
        std = self.sigmoid(self.fc_std(x)) + 1e-5
        return mean, std

    def choose_action(self, state):
        mean, std = self.forward(state)
        dis = Normal(mean, std)
        return dis.sample().numpy()


class CNNRotationPart(nn.Module):
    def __init__(self, action_dim):
        super(CNNRotationPart, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)  # 84 * 84 * 1 -> 82 * 82 * 32
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)  # 82 * 82 * 32 -> 41 * 41 * 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1)  # 41 * 41 * 32 -> 20 * 20 * 64
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(20 * 20 * 64, 100)
        self.relu3 = nn.ReLU()
        self.fc_mean_rotation = nn.Linear(100 + 1, action_dim)  # +3 for 3d position input
        self.tanh = nn.Tanh()
        self.fc_std_rotation = nn.Linear(100 + 1, action_dim)
        self.sigmoid = nn.Softplus()

        nn.init.kaiming_normal_(self.conv1.weight.data)
        nn.init.kaiming_normal_(self.conv2.weight.data)
        nn.init.kaiming_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc_mean_rotation.weight.data)
        nn.init.xavier_normal_(self.fc_std_rotation.weight.data)

    def forward(self, x, position):
        with torch.no_grad():
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.maxpool(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = self.fc1(x.view(x.size(0), -1))
            x = self.relu3(x)
        x = torch.cat((x, position), 1)
        mean = self.tanh(self.fc_mean_rotation(x))
        std = self.sigmoid(self.fc_std_rotation(x)) + 1e-5
        return mean, std

    def choose_action(self, state):
        mean, std = self.forward(state)
        dis = Normal(mean, std)
        return dis.sample().numpy()


class CNNRotationSup(nn.Module):
    def __init__(self, action_dim=1):
        super(CNNRotationSup, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1)  # 48 * 48 * 1 -> 44 * 44 * 32 224*224*1 -> 220*220*32
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)  # 44 * 44 * 32 -> 22 * 22 * 32 220*220*32 -> 110*110*32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 110 * 110 * 32 -> 110 * 110 * 64
        self.relu2 = nn.ReLU()

        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # 22 * 22 * 64 -> 22 * 22 * 64
        self.relu2_2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=1)  # 22 * 22 * 64 -> 18 * 18 * 64
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(18 * 18 * 64, 1024)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 64)
        self.relu5 = nn.ReLU()
        self.fc_mean = nn.Linear(64, action_dim)  # +3 for 3d position input
        self.tanh = nn.Tanh()

        nn.init.xavier_normal_(self.conv1.weight.data)
        nn.init.xavier_normal_(self.conv2.weight.data)
        nn.init.xavier_normal_(self.conv2_2.weight.data)
        nn.init.xavier_normal_(self.conv3.weight.data)
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.xavier_normal_(self.fc_mean.weight.data)

    def forward(self, x):
        # torch.set_printoptions(threshold=np.inf)
        x = self.conv1(x)
        # x.register_hook(lambda g: print('model x1 before gradient: ', g))
        x = self.relu1(x)
        # x.register_hook(lambda g: print('model x1 gradient: ', g))
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        # x.register_hook(lambda g: print('model x2 gradient: ', g))
        X = self.conv2_2(x)
        x = self.relu2_2(x)
        # x.register_hook(lambda g: print('model x3 gradient: ', g))
        x = self.conv3(x)
        x = self.relu3(x)
        # x.register_hook(lambda g: print('model x4 gradient: ', g))
        x = self.fc1(x.view(x.size(0), -1))
        x = self.relu4(x)
        # x.register_hook(lambda g: print('model x5 gradient: ', g))
        x = self.fc2(x)
        x = self.relu5(x)
        # x.register_hook(lambda g: print('model x6 gradient: ', g))
        mean = self.tanh(self.fc_mean(x))
        # mean.register_hook(lambda g: print('model mean gradient: ', g))
        mean = mean * np.pi
        return mean


class CNNRotationSup224(nn.Module):
    def __init__(self, action_dim=1):
        super(CNNRotationSup224, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=6, stride=2)  # 224 * 224 * 1 -> 110 * 110 * 32
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)  # 110 * 110 * 32 -> 55 * 55 * 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 55 * 55 * 32 -> 55 * 55 * 64
        self.relu2 = nn.ReLU()

        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # 55 * 55 * 64 -> 55 * 55 * 64
        self.relu2_2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=2)  # 55 * 55 * 64 -> 26 * 26 * 64
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(26 * 26 * 64, 1024)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 64)
        self.relu5 = nn.ReLU()
        self.fc_mean = nn.Linear(64, action_dim)  # +3 for 3d position input
        self.tanh = nn.Tanh()

        nn.init.xavier_normal_(self.conv1.weight.data)
        nn.init.xavier_normal_(self.conv2.weight.data)
        nn.init.xavier_normal_(self.conv2_2.weight.data)
        nn.init.xavier_normal_(self.conv3.weight.data)
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.xavier_normal_(self.fc_mean.weight.data)

    def forward(self, x):
        # torch.set_printoptions(threshold=np.inf)
        x = self.conv1(x)
        # x.register_hook(lambda g: print('model x1 before gradient: ', g))
        x = self.relu1(x)
        # x.register_hook(lambda g: print('model x1 gradient: ', g))
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        # x.register_hook(lambda g: print('model x2 gradient: ', g))
        X = self.conv2_2(x)
        x = self.relu2_2(x)
        # x.register_hook(lambda g: print('model x3 gradient: ', g))
        x = self.conv3(x)
        x = self.relu3(x)
        # x.register_hook(lambda g: print('model x4 gradient: ', g))
        x = self.fc1(x.view(x.size(0), -1))
        x = self.relu4(x)
        # x.register_hook(lambda g: print('model x5 gradient: ', g))
        x = self.fc2(x)
        x = self.relu5(x)
        # x.register_hook(lambda g: print('model x6 gradient: ', g))
        mean = self.tanh(self.fc_mean(x))
        # mean.register_hook(lambda g: print('model mean gradient: ', g))
        mean = mean * np.pi
        return mean

