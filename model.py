import torch.nn as nn
from torch.distributions import Normal
import torch


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


class CNNPolicy(nn.Module):
    def __init__(self, action_dim):
        super(CNNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)  # 84 * 84 * 1 -> 82 * 82 * 32
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)  # 82 * 82 * 32 -> 41 * 41 * 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1)  # 41 * 41 * 32 -> 20 * 20 * 64
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(20 * 20 * 64, 100)
        self.relu3 = nn.ReLU()
        self.fc_mean = nn.Linear(100 + 3, action_dim)  # +3 for 3d position input
        self.tanh = nn.Tanh()
        self.fc_std = nn.Linear(100 + 3, action_dim)
        self.sigmoid = nn.Softplus()

        nn.init.kaiming_normal_(self.conv1.weight.data)
        nn.init.kaiming_normal_(self.conv2.weight.data)
        nn.init.kaiming_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc_mean.weight.data)
        nn.init.xavier_normal_(self.fc_std.weight.data)

    def forward(self, x, position):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.fc1(x.view(x.size(0), -1))
        x = self.relu3(x)
        x = torch.cat((x, position), 1)
        mean = self.tanh(self.fc_mean(x))
        std = self.sigmoid(self.fc_std(x)) + 1e-5
        return mean, std

    def choose_action(self, state):
        mean, std = self.forward(state)
        dis = Normal(mean, std)
        return dis.sample().numpy()


class CNNValue(nn.Module):
    def __init__(self):
        super(CNNValue, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)  # 84 * 84 * 1 -> 82 * 82 * 32
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)  # 82 * 82 * 32 -> 41 * 41 * 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=1)  # 41 * 41 * 32 -> 20 * 20 * 64
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(20 * 20 * 64, 100)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(100 + 3, 1)    # +3 for 3d position input
        self.tanh = nn.Tanh()

        nn.init.kaiming_normal_(self.conv1.weight.data)
        nn.init.kaiming_normal_(self.conv2.weight.data)
        nn.init.kaiming_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)

    def forward(self, x, position):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.fc1(x.view(x.size(0), -1))
        x = self.relu3(x)
        # print('cnn forward x shape : ', x.shape)
        x = torch.cat((x, position), 1)
        x = self.fc2(x)

        return x



