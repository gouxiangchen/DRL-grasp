from VisualGrasp.environment import EnvGrasp
from VisualGrasp.model import CNNPolicy, CNNValue
from VisualGrasp.train import GraspTrain
from itertools import count
from MyDQN.logger import Logger


env = EnvGrasp()

gt = GraspTrain(CNNPolicy, CNNValue, 2, 2)
k = 0

log = Logger('./logs_grasp_cnn')
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
        action = gt.select_action_cnn(frame, pos)
        # print('action : ', action)
        reward, next_state, done, next_frame, next_pos = env.step(action)
        episode_reward += reward
        gt.buffer.add((state, action, next_state, reward, done, frame, next_frame, pos, next_pos))
        # print('reward : ', reward)
        if k > 32 or done == 1:
            k = 0
            gt.learn()
        if done == 1:
            break
        state = next_state
        frame = next_frame
        pos = next_pos
    gt.buffer.clear()
    if t % 10 == 0:
        print('in epoch ' + str(t) + '  episode reward : ', episode_reward)
    if t % 1000 == 999:
        gt.save_model('policy_cnn_only_grasp_' + str(t) + '.para', 'value_cnn_only_grasp_' + str(t) + '.para')
    info = {'episode reward': episode_reward}
    for tag, value in info.items():
        log.scalar_summary(tag, value, step=steps)
