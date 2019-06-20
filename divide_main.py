from VisualGrasp.divide_environment import EnvGrasp
from VisualGrasp.divide_model import CNNPolicy, MLPPolicy, MLPValue, CNNRotation, MLPRotation, MLPRotationValue
from VisualGrasp.divide_train import GraspTrain
from itertools import count
from tensorboardX import SummaryWriter
import time
import numpy as np

env = EnvGrasp()

gt = GraspTrain(MLPPolicy, MLPValue, MLPRotation, MLPRotationValue, CNNPolicy, CNNRotation, action_dim=3, state_dim=3)
k = 0

# gt.load_model('policy_mlp_grasp_move_first.para', 'value_mlp_grasp_move_first.para', 'policy_cnn_grasp_move_first.para', 'value_cnn_grasp_move_first.para')

writer = SummaryWriter('./logs_grasp_divide_rate')
steps = 0
for t in count():
    steps += 1
    episode_reward = [0, 0]
    state, frame, pos = env.reset()
    # print(env.grasp())
    # time.sleep(100)
    # env.open_griper()
    # print('init state', state)
    for i in range(200):
        k += 1
        # print(state)
        # time.sleep(2)
        action = gt.select_action(state[0:2])
        rotation_action = gt.select_orientation([state[-1]])
        # print('action : ', action, 'rotation : ', rotation_action)
        action = np.append(action, rotation_action)
        # print('action : ', action)
        reward, next_state, done, next_frame, next_pos = env.step(action)
        # print(episode_reward, reward)
        episode_reward[0] += reward[0]
        episode_reward[1] += reward[1]
        # print(reward)
        # time.sleep(1)
        gt.buffer.add((state, action[0:2], rotation_action, next_state, reward, done, frame, next_frame, pos, next_pos))
        # print('reward : ', reward)
        if k > 32 or done == 1:
            gt.learn()
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
        gt.save_model('policy_mlp_grasp_divide_' + str(t) + '.para', 'value_mlp_grasp_divide_' + str(t) + '.para',
                      'orientation_mlp_divide_' + str(t) + '.para', 'orientation_mlp_value_divide_' + str(t) + '.para',
                      'policy_cnn_grasp_divide_' + str(t) + '.para', 'orientation_cnn_grasp_divide_' + str(t) + '.para')
    # writer.add_scalar('episode_reward', episode_reward, steps)
    writer.add_scalars('episode_reward', {'distance reward': episode_reward[0], 'rotation reward': episode_reward[1]}, steps)
