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


class GraspTrain(object):
    def __init__(self, mlp_policy_model, mlp_value_model, mlp_orientation, mlp_orientation_value,
                 cnn_policy_model, cnn_orientation, action_dim, state_dim,
                 lr=0.0001, update_steps=4, buff_capability=32, batch_size=32, gamma=0.9, ppo_epoch=10):
        self.mlp_policy = mlp_policy_model(state_dim-1, action_dim-1).cuda()
        self.mlp_old_policy = mlp_policy_model(state_dim-1, action_dim-1).cuda()
        self.mlp_value = mlp_value_model(state_dim-1).cuda()

        self.mlp_old_orientation = mlp_orientation(1, 1).cuda()
        self.mlp_orientation = mlp_orientation(1, 1).cuda()
        self.mlp_orientation_value = mlp_orientation_value(1).cuda()

        self.cnn_policy = cnn_policy_model(action_dim-1).cuda()
        self.cnn_orientation = cnn_orientation(1).cuda()

        self.steps = 0
        self.buffer = Memory(buff_capability)
        self.buffer_capability = buff_capability
        self.buffer_count = 0
        self.update_steps = update_steps
        self.gamma = gamma
        self.policy_optim = torch.optim.Adam(self.mlp_policy.parameters(), lr=1e-4)
        self.valu_optim = torch.optim.Adam(self.mlp_value.parameters(), lr=2e-4)

        self.orientation_optim = torch.optim.Adam(self.mlp_orientation.parameters(), lr=1e-4)
        self.orientation_value_optim = torch.optim.Adam(self.mlp_orientation_value.parameters(), lr=2e-4)

        self.cnn_policy_optim = torch.optim.Adam(self.cnn_policy.parameters(), lr=1e-4)
        self.cnn_orientation_optim = torch.optim.Adam(self.cnn_orientation.parameters(), lr=1e-3)

        # self.optimizer2 = torch.optim.Adam([{'params': self.mlp_policy.parameters(), 'lr': 1e-4},
        #                                     {'params': self.mlp_value.parameters(), 'lr': 2e-4}])

        self.loss_steps = 0
        self.loss_mlp_steps = 0
        self.orientation_mlp_steps = 0
        self.orientation_steps = 0
        self.ppo_epoch = ppo_epoch
        self.cnn_epoch = 30
        self.batch_size = batch_size

        self.writer = SummaryWriter('./logs_grasp_divide')
        # self.logger = Logger('./logs_grasp')

        # self.mlplogger = Logger('./logs_grasp')

    def learn(self):
        self.steps += 1
        experiences = self.buffer.sample(self.batch_size)
        self.mlp_old_policy.load_state_dict(self.mlp_policy.state_dict())
        self.mlp_old_orientation.load_state_dict(self.mlp_orientation.state_dict())

        # state , action, orientation_action, next_state, reward, done, observation, next_observation, current_position, next_position
        state, action, orientation_action, next_state, reward, done, ob, next_ob, pos, next_pos = zip(*experiences)
        state = np.asarray(state)
        next_state = np.asarray(next_state)
        reward = np.asarray(reward)
        pos = np.asarray(pos)
        orientation_action = np.asarray(orientation_action)

        # print(state.shape, orientation_action.shape)
        orientation = state[:, -1]
        state = state[:, 0:2]
        next_orientation = next_state[:, -1]
        next_state = next_state[:, 0:2]
        orientation_reward = reward[:, -1]
        reward = reward[:, 0]
        orientation_pos = pos[:, -1]
        pos = pos[:, 0:2]

        orientation = torch.FloatTensor(orientation).cuda().unsqueeze(1)
        state = torch.FloatTensor(state).cuda()
        action = torch.FloatTensor(action).cuda()
        orientation_action = torch.FloatTensor(orientation_action).cuda().unsqueeze(1)
        next_state = torch.FloatTensor(next_state).cuda()
        next_orientation = torch.FloatTensor(next_orientation).cuda().unsqueeze(1)
        orientation_reward = torch.FloatTensor(orientation_reward).cuda().unsqueeze(1)
        reward = torch.FloatTensor(reward).cuda().unsqueeze(1)
        done = torch.FloatTensor(done).cuda().unsqueeze(1)

        ob = torch.FloatTensor(ob).cuda().unsqueeze(1)
        pos = torch.FloatTensor(pos).cuda()
        orientation_pos = torch.FloatTensor(orientation_pos).cuda().unsqueeze(1)

        # print(state.shape, action.shape, next_state.shape,
        #       reward.shape, ob.shape,
        #       pos.shape, orientation_pos.shape,
        #       orientation_reward.shape, orientation_action.shape, orientation.shape)
        # print(1-done)

        with torch.no_grad():
            old_mean, old_std = self.mlp_old_policy(state)
            old_normal = Normal(old_mean, old_std)

            target_v = reward + self.gamma * (1-done) * self.mlp_value(next_state)
            advantage = (target_v - self.mlp_value(state))
        # print('target v : ', target_v.shape)
        # print('advantage : ', advantage)

        for _ in range(self.ppo_epoch):
            self.loss_mlp_steps += 1
            mean, std = self.mlp_policy(state)
            n = Normal(mean, std)
            action_log_prob = n.log_prob(action)
            old_action_prob = old_normal.log_prob(action)
            # print('before : ', action_log_prob.shape, old_action_prob.shape)
            action_log_prob = torch.sum(action_log_prob, dim=1, keepdim=True)
            old_action_prob = torch.sum(old_action_prob, dim=1, keepdim=True)
            # print('position : ', action_log_prob.shape, old_action_prob.shape)
            ratio = torch.exp(action_log_prob - old_action_prob)
            # print(ratio)
            # ratio = torch.mean(ratio, 1, keepdim=True)
            # print(ratio.shape)
            # print('advantage [index] : ', advantage[index].shape)
            L1 = ratio * advantage
            # print(ratio)
            L2 = torch.clamp(ratio, 0.8, 1.2) * advantage
            action_loss = torch.min(L1, L2)  # + 1e-3 * n.entropy()
            action_loss = - action_loss.mean()

            value_loss = F.mse_loss(self.mlp_value(state), target_v)

            self.policy_optim.zero_grad()
            action_loss.backward()
            self.policy_optim.step()
            self.valu_optim.zero_grad()
            value_loss.backward()
            self.valu_optim.step()

            self.writer.add_scalar('action loss mlp', action_loss.item(), self.loss_mlp_steps)
            self.writer.add_scalar('value loss mlp', value_loss.item(), self.loss_mlp_steps)
            #
            # self.optimizer2.zero_grad()
            # action_loss.backward()
            # value_loss.backward()
            # self.optimizer2.step()

        with torch.no_grad():
            old_mean, old_std = self.mlp_old_orientation(orientation)
            old_normal = Normal(old_mean, old_std)

            target_v = orientation_reward + self.gamma * (1-done) * self.mlp_orientation_value(next_orientation)
            advantage = (target_v - self.mlp_orientation_value(orientation))
        # print('target v : ', target_v.shape)
        # print('advantage : ', advantage)

        for _ in range(self.ppo_epoch):
            self.orientation_mlp_steps += 1
            mean, std = self.mlp_orientation(orientation)
            n = Normal(mean, std)
            action_log_prob = n.log_prob(orientation_action)
            old_action_prob = old_normal.log_prob(orientation_action)
            # print('before : ', action_log_prob.shape, old_action_prob.shape)
            action_log_prob = torch.sum(action_log_prob, dim=1, keepdim=True)
            old_action_prob = torch.sum(old_action_prob, dim=1, keepdim=True)
            # print(action_log_prob.shape, old_action_prob.shape)
            ratio = torch.exp(action_log_prob - old_action_prob)
            # print(ratio)
            # ratio = torch.mean(ratio, 1, keepdim=True)
            # print(ratio.shape)
            # print('advantage [index] : ', advantage[index].shape)
            L1 = ratio * advantage
            # print(ratio)
            L2 = torch.clamp(ratio, 0.8, 1.2) * advantage
            action_loss = torch.min(L1, L2)  # + 1e-3 * n.entropy()
            action_loss = - action_loss.mean()

            value_loss = F.mse_loss(self.mlp_orientation_value(orientation), target_v)

            self.orientation_optim.zero_grad()
            action_loss.backward()
            self.orientation_optim.step()
            self.orientation_value_optim.zero_grad()
            value_loss.backward()
            self.orientation_value_optim.step()

            self.writer.add_scalar('orientation loss mlp', action_loss.item(), self.orientation_mlp_steps)
            self.writer.add_scalar('orientation value loss mlp', value_loss.item(), self.orientation_mlp_steps)
            #
            # self.optimizer2.zero_grad()
            # action_loss.backward()
            # value_loss.backward()
            # self.optimizer2.step()


        with torch.no_grad():
            old_mean, old_std = self.mlp_policy(state)
            old_normal = Normal(old_mean, old_std)
            # print('position \n old mean: ', old_mean, '\n old std: ', old_std)
        for _ in range(self.cnn_epoch):
            self.loss_steps += 1
            mean, std = self.cnn_policy(ob, pos)
            n = Normal(mean, std)
            # print('position \n new mean: ', mean, '\n new std: ', std)
            kl_loss = torch.distributions.kl.kl_divergence(n, old_normal)
            # print('position : ', kl_loss)
            kl_loss = kl_loss.sum(dim=1, keepdim=True)

            # print(kl_loss.shape)
            kl_loss = kl_loss.mean()
            # print(kl_loss)
            # kl_loss.register_hook(lambda g: self.writer.add_scalar('position gradient ', g.item(), self.orientation_steps))

            self.cnn_policy_optim.zero_grad()
            kl_loss.backward()
            self.cnn_policy_optim.step()
            self.writer.add_scalar('action loss', kl_loss.item(), self.loss_steps)

            # info = {'action loss': action_loss.item(), 'value loss': value_loss.item(),
            #         'distribution ratio': ratio.mean().item()}
            # for tag, value in info.items():
            #     self.logger.scalar_summary(tag, value, step=self.loss_steps)
        with torch.no_grad():
            old_mean, old_std = self.mlp_orientation(orientation)
            old_normal = Normal(old_mean, old_std)
            # print('orientation \n old mean: ', old_mean, '\n old std: ', old_std)
        for _ in range(self.cnn_epoch):
            self.orientation_steps += 1
            mean, std = self.cnn_orientation(ob, orientation_pos)
            n = Normal(mean, std)
            # print('orientation \n new mean: ', mean, '\n new std: ', std)
            kl_loss = torch.distributions.kl.kl_divergence(n, old_normal)
            # print('orientation : ', kl_loss)
            kl_loss = kl_loss.sum(dim=1, keepdim=True)
            # print(kl_loss.shape)
            kl_loss = kl_loss.mean()
            # print(kl_loss)
            # kl_loss.register_hook(lambda g: self.writer.add_scalar('orientation gradient ', g.item(), self.orientation_steps))
            self.cnn_orientation_optim.zero_grad()
            kl_loss.backward()
            self.cnn_orientation_optim.step()
            self.writer.add_scalar('orientation action loss', kl_loss.item(), self.orientation_steps)

            # info = {'action loss': action_loss.item(), 'value loss': value_loss.item(),
            #         'distribution ratio': ratio.mean().item()}
            # for tag, value in info.items():
            #     self.logger.scalar_summary(tag, value, step=self.loss_steps)

    def learn_orientation(self):
        self.steps += 1
        experiences = self.buffer.sample(self.batch_size)
        self.mlp_old_orientation.load_state_dict(self.mlp_orientation.state_dict())

        # state , action, orientation_action, next_state, reward, done, observation, next_observation, current_position, next_position
        state, action, orientation_action, next_state, reward, done, ob, next_ob, pos, next_pos = zip(*experiences)
        state = np.asarray(state)
        next_state = np.asarray(next_state)
        reward = np.asarray(reward)
        pos = np.asarray(pos)
        orientation_action = np.asarray(orientation_action)

        # print(state.shape, orientation_action.shape)
        orientation = state[:, -1]
        state = state[:, 0:2]
        next_orientation = next_state[:, -1]
        next_state = next_state[:, 0:2]
        orientation_reward = reward[:, -1]
        reward = reward[:, 0]
        orientation_pos = pos[:, -1]
        pos = pos[:, 0:2]

        orientation = torch.FloatTensor(orientation).cuda().unsqueeze(1)
        state = torch.FloatTensor(state).cuda()
        action = torch.FloatTensor(action).cuda()
        orientation_action = torch.FloatTensor(orientation_action).cuda().unsqueeze(1)
        next_state = torch.FloatTensor(next_state).cuda()
        next_orientation = torch.FloatTensor(next_orientation).cuda().unsqueeze(1)
        orientation_reward = torch.FloatTensor(orientation_reward).cuda().unsqueeze(1)
        reward = torch.FloatTensor(reward).cuda().unsqueeze(1)
        done = torch.FloatTensor(done).cuda().unsqueeze(1)

        ob = torch.FloatTensor(ob).cuda().unsqueeze(1)
        pos = torch.FloatTensor(pos).cuda()
        orientation_pos = torch.FloatTensor(orientation_pos).cuda().unsqueeze(1)

        with torch.no_grad():
            old_mean, old_std = self.mlp_old_orientation(orientation)
            old_normal = Normal(old_mean, old_std)

            target_v = orientation_reward + self.gamma * (1-done) * self.mlp_orientation_value(next_orientation)
            advantage = (target_v - self.mlp_orientation_value(orientation))
        # print('target v : ', target_v.shape)
        # print('advantage : ', advantage)

        for _ in range(self.ppo_epoch):
            self.orientation_mlp_steps += 1
            mean, std = self.mlp_orientation(orientation)
            n = Normal(mean, std)
            action_log_prob = n.log_prob(orientation_action)
            old_action_prob = old_normal.log_prob(orientation_action)
            # print('before : ', action_log_prob.shape, old_action_prob.shape)
            action_log_prob = torch.sum(action_log_prob, dim=1, keepdim=True)
            old_action_prob = torch.sum(old_action_prob, dim=1, keepdim=True)
            # print(action_log_prob.shape, old_action_prob.shape)
            ratio = torch.exp(action_log_prob - old_action_prob)
            # print(ratio)
            # ratio = torch.mean(ratio, 1, keepdim=True)
            # print(ratio.shape)
            # print('advantage [index] : ', advantage[index].shape)
            L1 = ratio * advantage
            # print(ratio)
            L2 = torch.clamp(ratio, 0.8, 1.2) * advantage
            action_loss = torch.min(L1, L2)  # + 1e-3 * n.entropy()
            action_loss = - action_loss.mean()

            value_loss = F.mse_loss(self.mlp_orientation_value(orientation), target_v)

            self.orientation_optim.zero_grad()
            action_loss.backward()
            self.orientation_optim.step()
            self.orientation_value_optim.zero_grad()
            value_loss.backward()
            self.orientation_value_optim.step()

            self.writer.add_scalar('orientation loss mlp', action_loss.item(), self.orientation_mlp_steps)
            self.writer.add_scalar('orientation value loss mlp', value_loss.item(), self.orientation_mlp_steps)
            #
            # self.optimizer2.zero_grad()
            # action_loss.backward()
            # value_loss.backward()
            # self.optimizer2.step()

        with torch.no_grad():
            old_mean, old_std = self.mlp_orientation(orientation)
            old_normal = Normal(old_mean, old_std)

        for _ in range(self.cnn_epoch):
            self.orientation_steps += 1
            mean, std = self.cnn_orientation(ob, orientation_pos)
            n = Normal(mean, std)

            kl_loss = torch.distributions.kl.kl_divergence(n, old_normal)
            # kl_loss.register_hook(lambda g: print('origin gradient: ', g))
            # print(kl_loss.shape)
            # kl_loss = kl_loss.sum(dim=1, keepdim=True)
            # print(kl_loss.shape)
            kl_loss = kl_loss.mean()
            # kl_loss.register_hook(lambda g: print('mean gradient: ', g))
            # print(kl_loss)
            # time.sleep(1)
            self.cnn_orientation_optim.zero_grad()
            kl_loss.backward()
            self.cnn_orientation_optim.step()
            self.writer.add_scalar('orientation action loss', kl_loss.item(), self.orientation_steps)

            # info = {'action loss': action_loss.item(), 'value loss': value_loss.item(),
            #         'distribution ratio': ratio.mean().item()}
            # for tag, value in info.items():
            #     self.logger.scalar_summary(tag, value, step=self.loss_steps)

    def select_action(self, state):
        state = torch.FloatTensor(state).cuda().unsqueeze(0)
        # print('state : ', state)
        with torch.no_grad():
            mean, std = self.mlp_policy(state)
        # print('mean, std', mean.shape, std.shape, mean, std)
        dist = Normal(mean, std)
        action = dist.sample()
        # action_log_prob = dist.log_prob(action)
        action = action.clamp(-1, 1)
        # print(action_log_prob.shape, action.shape)
        # print(action)
        return action.cpu().squeeze().numpy()
        # return action.cpu().numpy()

    def select_orientation(self, state):
        state = torch.FloatTensor(state).cuda().unsqueeze(0)
        # print('state : ', state)
        with torch.no_grad():
            mean, std = self.mlp_orientation(state)
        # print('mean, std', mean.shape, std.shape, mean, std)
        dist = Normal(mean, std)
        action = dist.sample()
        # action_log_prob = dist.log_prob(action)
        action = action.clamp(-1, 1)
        # print(action_log_prob.shape, action.shape)
        # print(action)
        return action.cpu().squeeze().numpy()
        # return action.cpu().numpy()

    def select_action_cnn(self, frame, pos):
        frame = torch.FloatTensor(frame).cuda().unsqueeze(0).unsqueeze(1)
        pos = torch.FloatTensor(pos).cuda().unsqueeze(0)
        # print('pos : ', pos, pos.shape)
        # print('state : ', state)
        with torch.no_grad():
            mean, std = self.cnn_policy(frame, pos)
        # print('mean, std', mean.shape, std.shape, mean, std)
        dist = Normal(mean, std)
        action = dist.sample()
        # action_log_prob = dist.log_prob(action)
        action = action.clamp(-1, 1)
        # print(action_log_prob.shape, action.shape)
        # print(action)
        return action.cpu().squeeze().numpy()
        # return action.cpu().numpy()

    def select_orientation_cnn(self, frame, pos):
        frame = torch.FloatTensor(frame).cuda().unsqueeze(0).unsqueeze(1)
        pos = torch.FloatTensor(pos).cuda().unsqueeze(0)
        # print('pos : ', pos, pos.shape)
        # print('state : ', state)
        with torch.no_grad():
            mean, std = self.cnn_orientation(frame, pos)
        # print('mean, std', mean.shape, std.shape, mean, std)
        dist = Normal(mean, std)
        action = dist.sample()
        # action_log_prob = dist.log_prob(action)
        action = action.clamp(-1, 1)
        # print(action_log_prob.shape, action.shape)
        # print(action)
        return action.cpu().squeeze().numpy()

    def save_model(self, policy_path, value_path, orientation, orientation_value, cnn_policy_path, cnn_orientation):
        save_path = '/home/chen/PycharmProjects/Reinforcement/VisualGrasp/model_divide/' + policy_path
        save_value = '/home/chen/PycharmProjects/Reinforcement/VisualGrasp/model_divide/' + value_path
        save_path_orientation = '/home/chen/PycharmProjects/Reinforcement/VisualGrasp/model_divide/' + orientation
        save_value_orientation = '/home/chen/PycharmProjects/Reinforcement/VisualGrasp/model_divide/' + orientation_value
        save_path_cnn = '/home/chen/PycharmProjects/Reinforcement/VisualGrasp/model_divide/' + cnn_policy_path
        save_orientation_cnn = '/home/chen/PycharmProjects/Reinforcement/VisualGrasp/model_divide/' + cnn_orientation
        torch.save(self.mlp_policy.state_dict(), save_path)
        torch.save(self.mlp_value.state_dict(), save_value)
        torch.save(self.mlp_orientation.state_dict(), save_path_orientation)
        torch.save(self.mlp_orientation_value.state_dict(), save_value_orientation)
        torch.save(self.cnn_policy.state_dict(), save_path_cnn)
        torch.save(self.cnn_orientation.state_dict(), save_orientation_cnn)
        print('model_divide saved in ' + str(time.asctime(time.localtime(time.time()))))

    def save_orientation(self, cnn_orientation):
        save_orientation_cnn = '/home/chen/PycharmProjects/Reinforcement/VisualGrasp/model_divide/' + cnn_orientation
        torch.save(self.cnn_orientation.state_dict(), save_orientation_cnn)
        print('model_divide saved in ' + str(time.asctime(time.localtime(time.time()))))

    def load_model(self, policy_path, value_path, orientation, orientation_value, cnn_policy_path, cnn_orientation):
        save_path = '/home/chen/PycharmProjects/Reinforcement/VisualGrasp/model_divide/' + policy_path
        save_value = '/home/chen/PycharmProjects/Reinforcement/VisualGrasp/model_divide/' + value_path
        save_path_orientation = '/home/chen/PycharmProjects/Reinforcement/VisualGrasp/model_divide/' + orientation
        save_value_orientation = '/home/chen/PycharmProjects/Reinforcement/VisualGrasp/model_divide/' + orientation_value
        save_path_cnn = '/home/chen/PycharmProjects/Reinforcement/VisualGrasp/model_divide/' + cnn_policy_path
        save_orientation_cnn = '/home/chen/PycharmProjects/Reinforcement/VisualGrasp/model_divide/' + cnn_orientation
        self.mlp_policy.load_state_dict(torch.load(save_path))
        self.mlp_value.load_state_dict(torch.load(save_value))
        self.mlp_orientation.load_state_dict(torch.load(save_path_orientation))
        self.mlp_orientation_value.load_state_dict(torch.load(save_value_orientation))

        self.cnn_policy.load_state_dict(torch.load(save_path_cnn))
        self.cnn_orientation.load_state_dict(torch.load(save_orientation_cnn))
        print(policy_path + ' model_divide is loaded!')

    def load_cnn_model(self, cnn_policy_path, cnn_orientation):
        save_path_cnn = '/home/chen/PycharmProjects/Reinforcement/VisualGrasp/model_divide/' + cnn_policy_path
        save_orientation_cnn = '/home/chen/PycharmProjects/Reinforcement/VisualGrasp/model_divide/' + cnn_orientation
        self.cnn_policy.load_state_dict(torch.load(save_path_cnn))
        self.cnn_orientation.load_state_dict(torch.load(save_orientation_cnn))
        print('cnn model loaded!')

    def load_part_model(self, cnn_pos_policy):
        save_path_cnn = '/home/chen/PycharmProjects/Reinforcement/VisualGrasp/model/' + cnn_pos_policy
        part_model_dict = torch.load(save_path_cnn)
        orientation_dict = self.cnn_orientation.state_dict()
        part_model_dict = {k: v for k, v in part_model_dict.items() if k in orientation_dict}
        orientation_dict.update(part_model_dict)
        self.cnn_orientation.load_state_dict(orientation_dict)
        print('part model loaded!')
