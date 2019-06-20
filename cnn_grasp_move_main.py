from VisualGrasp.environment_move_first import EnvGrasp
from VisualGrasp.move_model import CNNPolicy, CNNValue, MLPPolicy, MLPValue
from VisualGrasp.cnn_train_move import GraspTrain
from itertools import count
from tensorboardX import SummaryWriter


env = EnvGrasp()

gt = GraspTrain(MLPPolicy, MLPValue, CNNPolicy, CNNValue, action_dim=3, state_dim=3)
k = 0

# gt.load_model('policy_mlp_grasp_move_first.para', 'value_mlp_grasp_move_first.para', 'policy_cnn_grasp_move_first.para', 'value_cnn_grasp_move_first.para')

writer = SummaryWriter('./logs_grasp_move_rate')
steps = 0
for t in count():
    steps += 1
    episode_reward = 0
    state, frame, pos = env.reset()
    # print(env.grasp())
    # time.sleep(100)
    # env.open_griper()
    # print('init state', state)
    for i in range(200):
        k += 1
        # print(state)
        # time.sleep(2)
        action = gt.select_action(state)
        # print('action : ', action)
        reward, next_state, done, next_frame, next_pos = env.step(action)
        episode_reward += reward
        gt.buffer.add((state, action, next_state, reward, done, frame, next_frame, pos, next_pos))
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
    if t % 50 == 49:
        gt.save_model('policy_mlp_grasp_move_first_' + str(t) + '.para', 'value_mlp_grasp_move_first_' + str(t) + '.para',
                      'policy_cnn_grasp_move_first_' + str(t) + '.para', 'value_cnn_grasp_move_first_' + str(t) + '.para')
    writer.add_scalar('episode_reward', episode_reward, steps)
