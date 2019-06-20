import numpy as np
from tensorboardX import SummaryWriter
from MyDQN import vrep
import time
import random
import cv2 as cv


image_pix = 224  # 输入图像的维度 image_pix * image_pix 灰度图


class EnvGrasp(object):
    def __init__(self):
        self.total_success = 0
        self.total_try = 0
        self.writer = SummaryWriter('./logs_grasp_e2e')
        self.object_work_space = np.asarray([[-0.6, -0.4], [-0.1, 0.1], [0.01, 0.31]])

        self.work_space = np.asarray([[-0.75, -0.25], [-0.25, 0.25], [0.01, 0.31]])

        vrep.simxFinish(-1)
        self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP on port 19997

        self.current_state = []     # 2d

        self.target = []    # 3d
        self.target_orientation = []
        self.target_handle = 0

        self.current_handle = 0
        self.current_position = [0, 0, 0]   # 3d
        self.current_orientation = []

        self.pre_distance = 0
        self.pre_orientation_distance = 0

        self.correct_count = -1

        self.object_choose_int = 0

        if self.sim_client == -1:
            print('Failed to connect to simulation (V-REP remote API server). Exiting.')
            exit()
        else:
            print('Connected to simulation.')

    def get_state(self):
        sim_ret, self.current_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_target',
                                                                vrep.simx_opmode_blocking)
        sim_ret, self.current_position = vrep.simxGetObjectPosition(self.sim_client, self.current_handle, -1,
                                                                    vrep.simx_opmode_blocking)
        sim_ret, self.current_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.current_handle, -1, vrep.simx_opmode_blocking)
        sim_ret, self.target_handle = vrep.simxGetObjectHandle(self.sim_client, 'shape001', vrep.simx_opmode_blocking)
        sim_ret, self.target = vrep.simxGetObjectPosition(self.sim_client, self.target_handle, -1, vrep.simx_opmode_blocking)

        sim_ret, self.target_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.target_handle, -1, vrep.simx_opmode_blocking)

        self.target = np.asarray(self.target)
        self.current_position = np.asarray(self.current_position)

        if self.target_orientation[1] < 0:
            ratio = -1
        else:
            ratio = 1
        if self.object_choose_int == 1:
            orientation_dis = 0.
        else:
            orientation_dis = (self.current_orientation[1] - ratio * (np.pi/2 - abs(self.target_orientation[1]))) / (np.pi)
        self.current_state = self.current_position - self.target
        self.current_state = np.append(self.current_state, orientation_dis)
        # print(orientation_dis)
        # print('current position : ', self.current_position, 'target position : ', self.target, 'target orientation : ', self.target_orientation, 'current orientation : ', self.current_orientation)
        # print('current state : ', self.current_state)
        frame = self.get_sensor_data()
        # pos = self.current_position
        pos = np.append(self.current_position, self.current_orientation[1])
        return self.current_state, frame, pos

    def get_reward(self, is_down=False, action=[0, 0, 0]):
        done = 0
        reached = False
        if self.current_position[0] < self.work_space[0][0] or self.current_position[0] > self.work_space[0][1] \
                        or self.current_position[1] < self.work_space[1][0] or self.current_position[1] > self.work_space[1][1] \
                        or self.current_position[2] < self.work_space[2][0] or self.current_position[2] > self.work_space[2][1]:
            print('arm is out of workspace!')
            done = 1
            self.correct_count = 0
            return [-1, -1], done, reached
        # print('self.current_state : ', self.current_state)

        cs = self.current_state[0:2]
        distance = np.linalg.norm(cs)
        cs = np.append(cs, self.current_state[-1])

        orientation_distance = np.linalg.norm(self.current_state[-1])
        # print('distance : ', distance, ' predistance : ', self.pre_distance)
        # reward = -distance

        # distance = np.linalg.norm(origin_action[0:2])
        # print('distance : ', distance)

        reward = [-distance, -orientation_distance/1.5]
        # print(distance)
        total_distance = np.linalg.norm(cs)
        action = np.asarray(action)
        action_m = np.linalg.norm(action)
        # print('action m : ', action_m)
        if distance < self.pre_distance:
            reward[0] += 0.3
            # reward = 0
        else:
            reward[0] -= 0.3
        if orientation_distance < self.pre_orientation_distance:
            reward[1] += 0.3
        else:
            reward[1] -= 0.3
        if distance < 0.02:
            # print('distance reached !')
            # self.correct_count += 1
            reached = True
            reward[0] += 0.3
        else:
            # self.correct_count = 0
            pass
        if orientation_distance < 0.015:
            # print('orientation reached !')
            reward[1] += 0.5
        if action_m < 0.06 and is_down:
            # pass
            print('reached !')
            reward[0] += 0.5
            reward[1] += 0.5
            self.correct_count += 1
            reached = True
            # reward = 0
        else:
            # pass
            self.correct_count = 0
        # if orientation_distance + distance < 0.02:
        #     self.correct_count += 1
        #     reached = True
        # else:
        #     self.correct_count = 0
        if self.correct_count >= 2 and is_down:
            # grasp
            success = self.grasp()
            if success:
                print('grasp success ! ')
                reward[0] = 1
                reward[1] = 1
            else:
                print('grasp failed !')
                reward[0] = -1
                reward[1] = -1
                self.correct_count = 0
            done = 1
        self.pre_distance = distance
        self.pre_orientation_distance = orientation_distance
        return reward, done, reached

    def get_target_orientation(self):
        sim_ret, self.target_handle = vrep.simxGetObjectHandle(self.sim_client, 'shape001', vrep.simx_opmode_blocking)
        sim_ret, self.target_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.target_handle, -1,
                                                                         vrep.simx_opmode_blocking)
        # print('target orientation : ', self.target_orientation)
        if self.object_choose_int == 1:
            return np.pi / 2
        else:
            return self.target_orientation[1]

    def reset(self):
        self.correct_count = -1
        sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_target',
                                                                   vrep.simx_opmode_blocking)
        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(1)
        sim_ret, self.RG2_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip', vrep.simx_opmode_blocking)
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1,
                                                               vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.47, 0, 0.254),
                                   vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi / 2, 0, np.pi / 2),
                                      vrep.simx_opmode_blocking)
        while gripper_position[2] > 0.4:  # V-REP bug requiring multiple starts and stops to restart
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1,
                                                                   vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.47, 0, 0.254),
                                       vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi / 2, 0, np.pi / 2),
                                          vrep.simx_opmode_blocking)
        # print('started!')

        target_x = random.random() * (self.object_work_space[0][1] - self.object_work_space[0][0]) + self.object_work_space[0][0]
        target_y = random.random() * (self.object_work_space[1][1] - self.object_work_space[1][0]) + self.object_work_space[1][0]
        # target_z = random.random() * (self.work_space[2][1] - self.work_space[2][0]) + self.work_space[2][0]
        # target_x = -0.47
        # target_y = 0

        target_z = 0.025

        self.target = [target_x, target_y, target_z]

        orientation_y = -random.random() * np.pi + np.pi/2

        # print()
        # vrep.simxPauseCommunication(self.sim_client, False)
        object_orientation = [-np.pi/2, orientation_y, -np.pi/2]
        curr_mesh_file = '/home/chen/stl-model/surface_cars.obj'
        self.object_choose_int = random.randint(0, 1)
        if self.object_choose_int == 1:
            curr_mesh_file = '/home/chen/stl-model/cup_no.obj'
            object_orientation = [0, 0, -np.pi / 2]
        curr_shape_name = 'shape001'
        object_color = [0.4, 0.7, 1]
        ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = vrep.simxCallScriptFunction(self.sim_client,
                                                                                              'remoteApiCommandServer',
                                                                                              vrep.sim_scripttype_childscript,
                                                                                              'importShape',
                                                                                              [0, 0, 255, 0],
                                                                                              self.target + object_orientation + object_color,
                                                                                              [curr_mesh_file,
                                                                                               curr_shape_name],
                                                                                              bytearray(),
                                                                                              vrep.simx_opmode_blocking)
        self.target = np.asarray(self.target)
        self.open_griper()
        sim_ret, self.current_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_target',
                                                                vrep.simx_opmode_blocking)
        sim_ret, self.current_position = vrep.simxGetObjectPosition(self.sim_client, self.current_handle, -1,
                                                                    vrep.simx_opmode_blocking)
        if orientation_y < 0:
            ratio = -1
        else:
            ratio = 1
        # target_state = np.append(self.target[0:2], ratio * (np.pi/2 - abs(orientation_y))/(np.pi))
        # current_state = np.append(self.current_position[0:2], 0)
        self.pre_distance = np.linalg.norm(self.target[0:2] - self.current_position[0:2])
        if self.object_choose_int == 1:
            self.pre_orientation_distance = 0.
        else:
            self.pre_orientation_distance = np.linalg.norm(ratio * (np.pi/2 - abs(orientation_y))/(np.pi))
        self.current_state, frame, pos = self.get_state()
        # test_pos = np.append([0, 0, 0], orientation_y - np.pi/2)
        # self.move_to(test_pos)
        cs = self.current_state[0:2]

        cs = np.append(cs, self.current_state[-1])

        return cs, frame, pos

    def open_griper(self):
        gripper_motor_velocity = 0.5
        gripper_motor_force = 20
        sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint',
                                                               vrep.simx_opmode_blocking)
        sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle,
                                                                    vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity,
                                        vrep.simx_opmode_blocking)
        gripper_fully_opened = False
        while gripper_joint_position < 0.0536:  # Block until gripper is fully open
            sim_ret, new_gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle,
                                                                        vrep.simx_opmode_blocking)
            if new_gripper_joint_position <= gripper_joint_position:
                return gripper_fully_opened
            gripper_joint_position = new_gripper_joint_position
        gripper_fully_opened = True
        return gripper_fully_opened

    def move_to(self, tool_position):
        sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_target', vrep.simx_opmode_blocking)
        # UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, UR5_target_handle, -1,
                                                                  vrep.simx_opmode_blocking)
        sim_ret, UR5_target_orientation = vrep.simxGetObjectOrientation(self.sim_client, UR5_target_handle, -1, vrep.simx_opmode_blocking)
        # print(UR5_target_position)
        tool_orientation = tool_position[-1]
        tool_position = tool_position[0:3]
        if UR5_target_position[2] <= 0.074:
            tool_position[2] = 0
        move_direction = np.asarray(tool_position)
        move_magnitude = np.linalg.norm(move_direction)
        if move_magnitude == 0 or not move_magnitude == move_magnitude:
            move_step = [0, 0, 0]
            num_move_steps = 0
            # print('magnitude error~!', move_magnitude, tool_position)
        else:
            move_step = 0.005 * move_direction / move_magnitude
            num_move_steps = int(np.floor(move_magnitude / 0.005))

        rotation_step = 0.05 if (tool_orientation > 0) else -0.05
        num_rotation_steps = int(np.floor(tool_orientation / rotation_step))

        # print('move direction : ', move_direction, 'move magnitude : ', move_magnitude, 'move step : ', move_step, 'num : ', num_move_steps)

        for step_iter in range(max(num_move_steps, num_rotation_steps)):
            vrep.simxSetObjectPosition(self.sim_client, UR5_target_handle, -1, (UR5_target_position[0] +
                                                                                move_step[0] * min(step_iter,
                                                                                                   num_move_steps),
                                                                                UR5_target_position[1] +
                                                                                move_step[1] * min(step_iter,
                                                                                                   num_move_steps),
                                                                                UR5_target_position[2] +
                                                                                move_step[2] * min(step_iter,
                                                                                                   num_move_steps)),
                                       vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, UR5_target_handle, -1,
                                          (np.pi/2, UR5_target_orientation[1] + rotation_step * min(step_iter, num_rotation_steps), np.pi/2),
                                          vrep.simx_opmode_blocking)

    def move_position(self, tool_position):
        sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_target', vrep.simx_opmode_blocking)
        # UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, UR5_target_handle, -1,
                                                                  vrep.simx_opmode_blocking)
        is_down = False
        if UR5_target_position[2] <= 0.074:
            tool_position[2] = 0
            is_down = True
        move_direction = np.asarray(tool_position)
        move_magnitude = np.linalg.norm(move_direction)
        if move_magnitude == 0 or not move_magnitude == move_magnitude:
            move_step = [0, 0, 0]
            num_move_steps = 0
            # print('magnitude error~!', move_magnitude, tool_position)
        else:
            move_step = 0.005 * move_direction / move_magnitude
            num_move_steps = int(np.floor(move_magnitude / 0.005))
        for step_iter in range(num_move_steps):
            vrep.simxSetObjectPosition(self.sim_client, UR5_target_handle, -1, (UR5_target_position[0] +
                                                                                move_step[0] * min(step_iter,
                                                                                                   num_move_steps),
                                                                                UR5_target_position[1] +
                                                                                move_step[1] * min(step_iter,
                                                                                                   num_move_steps),
                                                                                UR5_target_position[2] +
                                                                                move_step[2] * min(step_iter,
                                                                                                   num_move_steps)),
                                       vrep.simx_opmode_blocking)
        return is_down

    def move_orientation(self, tool_orientation):
        sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_target', vrep.simx_opmode_blocking)
        sim_ret, UR5_target_orientation = vrep.simxGetObjectOrientation(self.sim_client, UR5_target_handle, -1,
                                                                        vrep.simx_opmode_blocking)
        rotation_step = 0.05 if (tool_orientation > 0) else -0.05
        num_rotation_steps = int(np.floor(tool_orientation / rotation_step))
        for step_iter in range(num_rotation_steps):
            vrep.simxSetObjectOrientation(self.sim_client, UR5_target_handle, -1,
                                          (np.pi/2, UR5_target_orientation[1] + rotation_step * min(step_iter, num_rotation_steps), np.pi/2),
                                          vrep.simx_opmode_blocking)

    def move_down(self, direction=1):
        sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_target', vrep.simx_opmode_blocking)
        # UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, UR5_target_handle, -1,
                                                                  vrep.simx_opmode_blocking)
        # print(UR5_target_position)
        # time.sleep(200)
        if direction == 1:
            if self.object_choose_int == 1:
                move_direction = np.asarray([0, 0, 0.044 - UR5_target_position[2]])
                move_magnitude = UR5_target_position[2] - 0.044
            else:
                move_direction = np.asarray([0, 0, 0.015 - UR5_target_position[2]])
                move_magnitude = UR5_target_position[2] - 0.015

        else:
            move_direction = np.asarray([0, 0, 0.301 - UR5_target_position[2]])
            move_magnitude = 0.301 - UR5_target_position[2]
        if move_magnitude == 0 or not move_magnitude == move_magnitude:
            move_step = [0, 0, 0]
            num_move_steps = 0
            print('magnitude error~!', move_magnitude)
        else:
            move_step = 0.005 * move_direction / move_magnitude
            num_move_steps = int(np.floor(move_magnitude / 0.005))
            # print('move_direction : ', move_direction, 'move_magnitude : ', move_magnitude)
            # print('move step : ', move_step, 'num : ', num_move_steps)

        for step_iter in range(num_move_steps):
            vrep.simxSetObjectPosition(self.sim_client, UR5_target_handle, -1, (UR5_target_position[0],
                                                                                UR5_target_position[1],
                                                                                UR5_target_position[2] +
                                                                                move_step[2] * (step_iter+1)),
                                                                                vrep.simx_opmode_blocking)

    def close_gripper(self):
        gripper_motor_velocity = -0.5
        gripper_motor_force = 100
        sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint',
                                                               vrep.simx_opmode_blocking)
        sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle,
                                                                    vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity,
                                        vrep.simx_opmode_blocking)
        gripper_fully_closed = False
        while gripper_joint_position > -0.047:  # Block until gripper is fully closed
            sim_ret, new_gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle,
                                                                            vrep.simx_opmode_blocking)
            # print(gripper_joint_position)
            if new_gripper_joint_position >= gripper_joint_position:
                return gripper_fully_closed
            gripper_joint_position = new_gripper_joint_position
        gripper_fully_closed = True
        return gripper_fully_closed

    def grasp(self):
        self.total_try += 1

        self.move_down()
        # time.sleep(0.5)
        self.close_gripper()
        self.move_down(-1)  # move up
        time.sleep(1)
        sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint',
                                                               vrep.simx_opmode_blocking)
        sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle,
                                                                    vrep.simx_opmode_blocking)
        if 0.055 >= gripper_joint_position >= -0.04:
            success = True
        else:
            success = False
        if success:
            self.total_success += 1

        accuracy = float(self.total_success) / self.total_try

        self.writer.add_scalar('grasp success rate', accuracy, self.total_try)

        return success

    def get_sensor_data(self, is_save=False):
        sim_ret, cam_handle = vrep.simxGetObjectHandle(self.sim_client, 'Vision_sensor_hand', vrep.simx_opmode_blocking)

        # Get color image from simulation
        sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, cam_handle, 0,
                                                                       vrep.simx_opmode_blocking)
        color_img = np.asarray(raw_image)
        # print(color_img.shape)
        color_img.shape = (resolution[1], resolution[0], 3)
        # print(color_img)
        color_img = color_img.astype(np.float) / 255

        color_img[color_img < 0] += 1

        # color_img *= 255
        color_img = np.fliplr(color_img)

        test_show_image = color_img * 255
        test_show_image = test_show_image.astype(np.uint8)

        test_show_image = cv.cvtColor(test_show_image, cv.COLOR_BGR2GRAY)

        resize_color = cv.resize(test_show_image, (image_pix, image_pix))
        # print(resize_color.shape)
        # cropped_color = resize_color[84:132, 78:126]
        cropped_color = resize_color
        if is_save:
            # print(test_show_image.shape)
            # test_show_image = Image.fromarray(test_show_image)
            # test_show_image.show()
            cv.imwrite('/home/chen/PycharmProjects/Reinforcement/VisualGrasp/image/' + str(time.time()) + '.png', cropped_color)


        # print(cropped_color.shape)
        cropped_color = np.asarray(cropped_color)
        # print(cropped_color.shape)
        # cv.imwrite('t.png', resize_color)
        cropped_color = cropped_color.astype(np.float) / 255.
        # print(cropped_color.shape)
        # np.set_printoptions(threshold=np.inf, suppress=True)
        # f = open('test.txt', 'w')
        # print(resize_color, file=f)
        # f.close()
        # print('ok')
        # time.sleep(5)
        # print(cropped_color.shape)
        return cropped_color

    def step(self, action):
        # action : [x_ratio, y_ratio] ~ [-1, 1]
        # origin_action = action
        move_magnitude = 0.05
        rotation_magnitude = np.pi/10

        move = action[0:2]
        move = np.append(move, -0.4)
        rotation = action[-1]
        # print('rotation : ', rotation)
        rotation = rotation * rotation_magnitude
        move = move * move_magnitude

        is_down = self.move_position(move)
        self.move_orientation(rotation)

        # action = np.append(move, rotation)
        # self.move_to(action)
        self.current_state, frame, pos = self.get_state()
        reward, done, reached = self.get_reward(is_down, [move[0], move[1], rotation])
        cs = self.current_state[0:2]
        cs = np.append(cs, self.current_state[-1])
        save_orientation = is_down and reached
        # print('is down : ', is_down, 'reached : ', reached)
        return reward, cs, done, frame, pos, save_orientation
