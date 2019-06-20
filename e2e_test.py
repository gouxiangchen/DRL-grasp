from VisualGrasp.e2e_enviroment import EnvGrasp
from VisualGrasp.e2e_model import CNNPolicy, MLPPolicy, MLPValue, CNNRotationSup224, MLPRotation, MLPRotationValue
from VisualGrasp.e2e_trainer import GraspTrain
from itertools import count
from tensorboardX import SummaryWriter
import time
import numpy as np

env = EnvGrasp()

gt = GraspTrain(MLPPolicy, MLPValue, MLPRotation, MLPRotationValue,
                CNNPolicy, CNNRotationSup224, action_dim=3, state_dim=3)
k = 0
gt.load_cnn_model('policy_cnn_grasp_e2e_399.para', 'orientation_cnn_grasp_e2e_4599.para', 'orientation_mlp_e2e_399.para')
# gt.load_model('policy_mlp_grasp_move_first.para', 'value_mlp_grasp_move_first.para', 'policy_cnn_grasp_move_first.para', 'value_cnn_grasp_move_first.para')

# writer = SummaryWriter('./logs_grasp_e2e')
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
    orientation_action = 0
    orientation_begin = False
    for i in range(200):
        k += 1
        action = gt.select_action_cnn(frame, pos[0:2])
        current_orientation = pos[-1]
        if orientation_begin:
            orientation = gt.get_target_orientation_cnn(frame)
            orientation = orientation.item()
            if orientation < 0:
                ratio = -1
            else:
                ratio = 1

            orientation_dis = (current_orientation - ratio * (np.pi / 2 - abs(orientation))) / np.pi

            print('???', orientation, env.get_target_orientation(), orientation_dis, current_orientation)
            orientation_action = gt.select_orientation([orientation_dis])
        action = np.append(action, orientation_action)
        # print('action : ', action)
        reward, next_state, done, next_frame, next_pos, save_orientation = env.step(action)
        if save_orientation:
            orientation_begin = True
        if done == 1:
            break
        state = next_state
        frame = next_frame
        pos = next_pos
    gt.buffer.clear()

    if t % 10 == 0:
        print('in epoch ' + str(t) + '  episode reward : ', episode_reward, 'orientation size : ', gt.orientation_buffer.size())
    if t % 200 == 199:
        gt.save_model('policy_mlp_grasp_e2e_' + str(t) + '.para', 'value_mlp_grasp_e2e_' + str(t) + '.para',
                      'orientation_mlp_e2e_' + str(t) + '.para', 'orientation_mlp_value_e2e_' + str(t) + '.para',
                      'policy_cnn_grasp_e2e_' + str(t) + '.para', 'orientation_cnn_grasp_e2e_' + str(t) + '.para')
    # writer.add_scalar('episode_reward', episode_reward, steps)
    # writer.add_scalars('episode_reward', {'distance reward': episode_reward[0], 'rotation reward': episode_reward[1]}, steps)
