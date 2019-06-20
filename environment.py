import numpy as np
from MyDQN.logger import Logger
from MyDQN import vrep
import time
import random
import cv2 as cv


image_pix = 84  # 输入图像的维度 image_pix * image_pix 灰度图


class EnvGrasp(object):
    def __init__(self):
        self.total_success = 0
        self.total_try = 0
        self.logger = Logger('./logs_grasp')
        self.work_space = np.asarray([[-0.7, -0.3], [-0.2, 0.2], [0.1, 0.4]])

        self.object_work_space = np.asarray([[-0.6, -0.4], [-0.1, 0.1], [0.01, 0.31]])

        vrep.simxFinish(-1)
        self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)  # Connect to V-REP on port 19997

        self.current_state = []     # 2d

        self.target = []    # 3d

        self.current_handle = 0
        self.current_position = [0, 0, 0]   # 3d

        self.pre_distance = 0

        self.correct_count = 0

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

        self.current_position = np.asarray(self.current_position)
        # print('current position : ', self.current_position, 'target position : ', self.target)
        self.current_state = self.current_position - self.target
        frame = self.get_sensor_data()
        return self.current_state[0:2], frame, self.current_position

    def get_reward(self):
        done = 0
        if self.current_position[0] < self.work_space[0][0] or self.current_position[0] > self.work_space[0][1] \
                        or self.current_position[1] < self.work_space[1][0] or self.current_position[1] > self.work_space[1][1] \
                        or self.current_position[2] < self.work_space[2][0] or self.current_position[2] > self.work_space[2][1]:
            print('arm is out of workspace!')
            done = 1
            self.correct_count = 0
            return -1, done
        distance = np.linalg.norm(self.current_state)
        # print('distance : ', distance, ' predistance : ', self.pre_distance)
        reward = -distance
        # print(distance)
        if distance < self.pre_distance:
            reward += 0.3
        else:
            pass
        if distance < 0.015:
            self.correct_count += 1
            print('reached !')
            reward += 0.5
        else:
            self.correct_count = 0
        if self.correct_count >= 5:
            # grasp
            success = self.grasp()
            if success:
                print('grasp success ! ')
                reward = 1
            else:
                print('grasp failed !')
                reward = -1
            self.correct_count = 0
            done = 1
        self.pre_distance = distance
        return reward, done

    def reset(self):
        sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_target',
                                                                   vrep.simx_opmode_blocking)
        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(1)
        sim_ret, self.RG2_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip', vrep.simx_opmode_blocking)
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1,
                                                               vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.47, 0, 0.3),
                                   vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi / 2, 0, np.pi / 2),
                                      vrep.simx_opmode_blocking)
        while gripper_position[2] > 0.4:  # V-REP bug requiring multiple starts and stops to restart
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1,
                                                                   vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.47, 0, 0.3),
                                       vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi / 2, 0, np.pi / 2),
                                          vrep.simx_opmode_blocking)
        # print('started!')

        target_x = random.random() * (self.object_work_space[0][1] - self.object_work_space[0][0]) + self.object_work_space[0][0]
        target_y = random.random() * (self.object_work_space[1][1] - self.object_work_space[1][0]) + self.object_work_space[1][0]
        # target_z = random.random() * (self.work_space[2][1] - self.work_space[2][0]) + self.work_space[2][0]
        target_z = 0.075

        self.target = [target_x, target_y, target_z]
        # print()
        # vrep.simxPauseCommunication(self.sim_client, False)

        # orientation_y = -random.random() * np.pi + np.pi / 2

        # print()
        # vrep.simxPauseCommunication(self.sim_client, False)
        # object_orientation = [-np.pi / 2, orientation_y, -np.pi / 2]

        object_orientation = [0, 0, 0]
        curr_mesh_file = '/home/chen/stl-model/cup.obj'
        curr_shape_name = 'shape001'
        object_color = [0, 0.6, 1]
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
        sim_ret, self.current_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_target',
                                                                vrep.simx_opmode_blocking)
        sim_ret, self.current_position = vrep.simxGetObjectPosition(self.sim_client, self.current_handle, -1,
                                                                    vrep.simx_opmode_blocking)
        self.pre_distance = np.linalg.norm(self.target[0:2] - self.current_position[0:2])
        self.current_state, frame, pos = self.get_state()
        return self.current_state, frame, pos

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
        # print(UR5_target_position)
        move_direction = np.asarray(tool_position)
        move_magnitude = np.linalg.norm(move_direction)
        if move_magnitude == 0 or not move_magnitude == move_magnitude:
            move_step = [0, 0, 0]
            num_move_steps = 0
            print('magnitude error~!', move_magnitude, tool_position)
        else:
            move_step = 0.005 * move_direction / move_magnitude
            num_move_steps = int(np.floor(move_magnitude / 0.005))

        # print('move direction : ', move_direction, 'move magnitude : ', move_magnitude, 'move step : ', move_step, 'num : ', num_move_steps)

        for step_iter in range(num_move_steps):
            vrep.simxSetObjectPosition(self.sim_client, UR5_target_handle, -1, (UR5_target_position[0] +
                                                                                move_step[0] * min(step_iter,
                                                                                                   num_move_steps),
                                                                                UR5_target_position[1] +
                                                                                move_step[1] * min(step_iter,
                                                                                                   num_move_steps),
                                                                                UR5_target_position[2]),
                                       vrep.simx_opmode_blocking)

    def move_down(self, direction=1):
        sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_target', vrep.simx_opmode_blocking)
        # UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, UR5_target_handle, -1,
                                                                  vrep.simx_opmode_blocking)
        # print(UR5_target_position)
        # time.sleep(200)
        if direction == 1:
            move_direction = np.asarray([0, 0, 0.039 - UR5_target_position[2]])
            move_magnitude = UR5_target_position[2] - 0.039
        else:
            move_direction = np.asarray([0, 0, 0.301 - UR5_target_position[2]])
            move_magnitude = 0.301 - UR5_target_position[2]
        if move_magnitude == 0 or not move_magnitude == move_magnitude:
            move_step = [0, 0, 0]
            num_move_steps = 0
            print('magnitude error~!', move_magnitude)
        else:
            move_step = 0.02 * move_direction / move_magnitude
            num_move_steps = int(np.floor(move_magnitude / 0.02))
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
        self.open_griper()
        self.move_down()
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
        info = {'grasp success rate': accuracy}
        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, step=self.total_try)

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


        if is_save:
            # print(test_show_image.shape)
            # test_show_image = Image.fromarray(test_show_image)
            # test_show_image.show()
            cv.imwrite('/home/chen/PycharmProjects/Reinforcement/VisualGrasp/image/' + 'grasp_success' + str(time.asctime(time.localtime(time.time()))) + '.png', test_show_image)

        resize_color = cv.resize(test_show_image, (image_pix, image_pix))

        resize_color = np.asarray(resize_color)
        # cv.imwrite('t.png', resize_color)
        resize_color = resize_color.astype(np.float) / 255.
        # np.set_printoptions(threshold=np.inf, suppress=True)
        # f = open('test.txt', 'w')
        # print(resize_color, file=f)
        # f.close()
        # print(resize_color.shape)
        return resize_color

    def step(self, action):
        # action : [x_ratio, y_ratio] ~ [-1, 1]
        move_magnitude = 0.05
        action = np.append(action, 0)
        action = action * move_magnitude
        self.move_to(action)
        self.current_state, frame, pos = self.get_state()
        reward, done = self.get_reward()
        return reward, self.current_state, done, frame, pos
