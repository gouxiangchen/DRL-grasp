from VisualGrasp.e2e_enviroment import EnvGrasp
from VisualGrasp.e2e_model import CNNPolicy, MLPPolicy, MLPValue, CNNRotationSup224, MLPRotation, MLPRotationValue, MLPValue1024, MLPPolicy1024
from VisualGrasp.e2e_trainer import GraspTrain
from itertools import count
from tensorboardX import SummaryWriter
import time
import numpy as np


env = EnvGrasp()

gt = GraspTrain(MLPPolicy, MLPValue, MLPRotation, MLPRotationValue,
                CNNPolicy, CNNRotationSup224, action_dim=3, state_dim=3)
k = 0

gt.load_cnn_policy('policy_cnn_grasp_e2e.para')
gt.load_mlp_orientation('orientation_mlp_e2e.para')

# gt.load_active_model('active_policy_199.para', 'active_value_199.para')

writer = SummaryWriter('./logs_grasp_e2e_active')
steps = 0
for t in count():
    steps += 1
    episode_reward = [0, 0]
    state, frame, pos = env.reset()
    # print(env.grasp())
    # time.sleep(100)
    # env.open_griper()
    # print('init state', state)
    is_save = False
    for i in range(100):
        k += 1
        # print(state)
        # time.sleep(2)
        # action = gt.select_action(state[0:2])
        rotation_action = gt.select_orientation([state[-1]])
        action = gt.select_action_active(frame, pos[0:2])
        # print('action : ', action, 'rotation : ', rotation_action)
        action = np.append(action, rotation_action)
        # print('action : ', action)
        reward, next_state, done, next_frame, next_pos, save_orientation = env.step(action)
        # print('pos : ', pos.shape, next_pos.shape)
        # print(frame.shape)
        # print(episode_reward, reward)
        episode_reward[0] += reward[0]
        episode_reward[1] += reward[1]
        # print(reward)
        # time.sleep(1)
        gt.buffer.add((state, action[0:2], rotation_action, next_state, reward, done, frame, next_frame, pos, next_pos))
        if k > 32 or done == 1:
            gt.learn_active()
            k = 0
        if done == 1:
            break
        state = next_state
        frame = next_frame
        pos = next_pos
    gt.buffer.clear()

    if t % 10 == 0:
        print('in epoch ' + str(t) + '  episode reward : ', episode_reward)
    if t % 200 == 199:
        gt.save_active_model('active_policy_' + str(t) + '.para', 'active_value_' + str(t) + '.para')
    # writer.add_scalar('episode_reward', episode_reward, steps)
    writer.add_scalars('episode_reward', {'distance reward': episode_reward[0]}, steps)
