from VisualGrasp.e2e_enviroment import EnvGrasp
from VisualGrasp.e2e_model import CNNPolicy, MLPPolicy, MLPValue, CNNRotationSup224, MLPRotation, MLPRotationValue
from VisualGrasp.e2e_trainer import GraspTrain
from itertools import count
from tensorboardX import SummaryWriter
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import random


def make_one_hot(data1, dim=10):
    return (np.arange(dim) == data1[:, None]).astype(np.integer)


i = np.asarray([random.randint(0, 1) for k in range(100)])
print(i.sum())
