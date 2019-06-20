from VisualGrasp.e2e_enviroment import EnvGrasp
from VisualGrasp.e2e_model import CNNPolicy, MLPPolicy, MLPValue, CNNRotationSup224, MLPRotation, MLPRotationValue
from VisualGrasp.e2e_trainer import GraspTrain
from itertools import count
from tensorboardX import SummaryWriter
import time
import numpy as np
import cv2 as cv


def save_frame(frame):
    test_show_image = frame * 255
    test_show_image = test_show_image.astype(np.uint8)

    resize_color = cv.resize(test_show_image, (224, 224))
    # print(resize_color.shape)
    # cropped_color = resize_color[84:132, 78:126]
    cropped_color = resize_color
    cv.imwrite('/home/chen/PycharmProjects/Reinforcement/VisualGrasp/image/aob' + str(time.time()) + '.png',
               cropped_color)


env = EnvGrasp()

gt = GraspTrain(MLPPolicy, MLPValue, MLPRotation, MLPRotationValue,
                CNNPolicy, CNNRotationSup224, action_dim=3, state_dim=3)
k = 0

gt.load_model_('policy_mlp_grasp_e2e_399.para', 'value_mlp_grasp_e2e_399.para', 'orientation_mlp_e2e_399.para',
               'orientation_mlp_value_e2e_399.para', 'policy_cnn_grasp_e2e_399.para')
# gt.load_cnn_model('policy_cnn_grasp_e2e.para', 'orientation_cnn_grasp_e2e_999.para', 'orientation_mlp_e2e.para')

writer = SummaryWriter('./logs_grasp_e2e')
steps = 0
s = 0
for t in count():
    steps += 1
    episode_reward = [0, 0]
    state, frame, pos = env.reset()
    # print(env.grasp())
    # time.sleep(100)
    # env.open_griper()
    # print('init state', state)
    is_save = False
    for i in range(200):
        k += 1
        s += 1
        # print(state)
        # time.sleep(2)
        action = gt.select_action(state[0:2])
        rotation_action = gt.select_orientation([state[-1]])

        # print('action : ', action, 'rotation : ', rotation_action)
        action = np.append(action, rotation_action)
        # print('action : ', action)
        reward, next_state, done, next_frame, next_pos, save_orientation = env.step(action)
        # print(frame.shape)
        # print(episode_reward, reward)
        episode_reward[0] += reward[0]
        episode_reward[1] += reward[1]
        # print(reward)
        # time.sleep(1)
        writer.add_scalars('step_reward',
                           {'distance reward': reward[0], 'rotation reward': reward[1]}, s)
        gt.buffer.add((state, action[0:2], rotation_action, next_state, reward, done, frame, next_frame, pos, next_pos))
        if save_orientation and not is_save:
            label = env.get_target_orientation()
            gt.orientation_buffer.add((frame, label))
            is_save = True
            print('orientation size : ', gt.orientation_buffer.size())
            orientation_eval = gt.get_orientation_eval(frame)
            print('label : ', label, 'eval : ', orientation_eval)
            # print(frame.shape)
            # save_frame(frame)
        # print('reward : ', reward)
        if gt.orientation_buffer.size() > 320:
            pass
            gt.learn_orientation()
        if k > 32 or done == 1:
            # gt.learn()
            k = 0
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
    writer.add_scalars('episode_reward', {'distance reward': episode_reward[0], 'rotation reward': episode_reward[1]}, steps)
