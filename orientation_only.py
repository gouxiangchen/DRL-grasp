from VisualGrasp.divide_environment import EnvGrasp
from VisualGrasp.divide_model import CNNPolicy, MLPPolicy, MLPValue, CNNRotation, MLPRotation, MLPRotationValue, CNNRotationBig, CNNRotationPart
from VisualGrasp.divide_train import GraspTrain
from itertools import count
from tensorboardX import SummaryWriter
import time
import numpy as np

env = EnvGrasp()

gt = GraspTrain(MLPPolicy, MLPValue, MLPRotation, MLPRotationValue, CNNPolicy, CNNRotationPart,
                action_dim=3, state_dim=3, batch_size=32, buff_capability=32)
k = 0

# gt.load_model('policy_mlp_grasp_move_first.para', 'value_mlp_grasp_move_first.para', 'policy_cnn_grasp_move_first.para', 'value_cnn_grasp_move_first.para')
gt.load_part_model('policy_cnn_grasp.para')
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
        action = [0, 0]
        # action = gt.select_action(state[0:2])
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
            gt.learn_orientation()
            k = 0
        if done == 1:
            break
        state = next_state
        frame = next_frame
        pos = next_pos
    gt.buffer.clear()

    if t % 10 == 0:
        print('in epoch ' + str(t) + '  episode reward : ', episode_reward)
    if t % 100 == 99:
        gt.save_orientation('cnn_orientation_only_' + str(t) + '_.para')
    # writer.add_scalar('episode_reward', episode_reward, steps)
    writer.add_scalars('episode_reward', {'distance reward': episode_reward[0], 'rotation reward': episode_reward[1]}, steps)
