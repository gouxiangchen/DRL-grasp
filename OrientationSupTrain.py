import numpy as np
import random
import torch

import time

from torch.distributions import Normal
from collections import deque
import torch.nn.functional as F
from tensorboardX import SummaryWriter


class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()


class SupTrain(object):
    def __init__(self, sup_orientation_model):
        self.model = sup_orientation_model().cuda()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.batch_size = 32
        self.buffer = Memory(1000)
        self.buffer_count = 0
        self.steps = 0
        self.writer = SummaryWriter('./logs_orientation_sup')

    def learn(self):

        experiences = self.buffer.sample(self.batch_size, continuous=False)
        frames, labels = zip(*experiences)
        frames = torch.FloatTensor(frames).cuda().unsqueeze(1)
        labels = torch.FloatTensor(labels).cuda().unsqueeze(1)

        # print(frames.shape, labels.shape)
        for _ in range(1):
            self.steps += 1
            out = self.model(frames)

            # print('out: ', out.cpu().numpy(), 'label: ', labels.cpu().numpy)
            loss = F.mse_loss(labels, out)
            print(loss)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.writer.add_scalar('loss', loss.item(), self.steps)

    def get_out(self, frame):
        frame = torch.FloatTensor(frame).unsqueeze(0).unsqueeze(0).cuda()
        # print(frame.shape)
        out = self.model(frame)
        return out

    def save_model(self, path):
        save_path = '/home/chen/PycharmProjects/Reinforcement/VisualGrasp/model_orientation_sup/' + path
        torch.save(self.model.state_dict(), save_path)
        print('model saved in ' + save_path)

    def load_model(self, path):
        save_path = '/home/chen/PycharmProjects/Reinforcement/VisualGrasp/model_orientation_sup/' + path
        self.model.load_state_dict(torch.load(save_path))
        print('model loaded!')
