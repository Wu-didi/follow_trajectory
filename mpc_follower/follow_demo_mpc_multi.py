import os
import sys
import time
import math
import threading
import logging
from math import radians, cos, sin, asin, sqrt, degrees, atan2

import can
import cvxpy
import numpy as np
from pyproj import CRS, Transformer
from pynput import keyboard
from read_csv import read_csv  # 确保此模块存在并正确导入
import matplotlib.pyplot as plt

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 投影转换器初始化
wgs84 = CRS("EPSG:4326")
utm_zone_number = 50  # 根据实际情况选择合适的 UTM 区域
utm_crs = CRS(f"EPSG:{32600 + utm_zone_number}")  # 例如，UTM Zone 50N 对应 EPSG:32650
projector_to_utm = Transformer.from_crs(wgs84, utm_crs, always_xy=True)
projector_to_wgs84 = Transformer.from_crs(utm_crs, wgs84, always_xy=True)


def latlon_to_utm(lon, lat):
    """将经纬度转换为 UTM 坐标"""
    x, y = projector_to_utm.transform(lon, lat)
    return x, y


def smooth_yaw_iter(previous_yaw, new_yaw):
    """
    Smooth the yaw angle based on the previous yaw to ensure continuity.
    """
    dyaw = new_yaw - previous_yaw
    dyaw = (dyaw + np.pi) % (2.0 * np.pi) - np.pi
    smoothed_yaw = previous_yaw + dyaw
    return smoothed_yaw


def angle_mod(x, zero_2_2pi=False, degree=False):
    """
    Angle modulo operation
    Default angle modulo range is [-pi, pi)
    """
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle


def pi_2_pi(angle):
    return angle_mod(angle)


def get_nparray_from_matrix(x):
    return np.array(x).flatten()


def get_linear_model_matrix(v, phi, delta):
    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = -DT * v * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB

    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(phi) * phi
    C[1] = -DT * v * math.cos(phi) * phi
    C[3] = -DT * v * delta / (WB * math.cos(delta) ** 2)

    return A, B, C


# 车辆参数
VEHICLE_WIDTH = 1.9   # m
VEHICLE_LENGTH = 4.5  # m
WHEEL_FACTOR = 7.2
manual_triggered = False
stop_record = False
mod_666 = 0
mod_AE = 0

# MPC参数
NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
T = 5  # horizon length

# MPC成本矩阵
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 1.0])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.5, 0.5])  # state cost matrix
Qf = Q  # state final matrix
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 500.0  # max simulation time

# 迭代参数
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param

TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number

DT = 0.5  # [s] time tick

# Vehicle parameters
LENGTH = 4.5  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 2.5  # [m]

MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]

show_animation = True


class State:
    """
    Vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None


def calc_speed_profile(cx, cy, cyaw, target_speed):
    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = -target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile


class ISGSpeedFilter:
    def __init__(self):
        self.isg_sum_filtspd = 0  # 总和
        self.isg_mot_spd_filt = 0  # 滤波后的速度
        self.isg_mot_spd = 0  # 实时速度
        self.MAT_Moto_spd = 0  # 最终输出的速度

    def update_speed(self, isg_mot_spd):
        self.isg_mot_spd = isg_mot_spd
        self.isg_sum_filtspd += self.isg_mot_spd  # 加上当前速度
        self.isg_sum_filtspd -= self.isg_mot_spd_filt  # 减去上一次的滤波结果

        # 计算滤波后的速度
        self.isg_mot_spd_filt = self.isg_sum_filtspd / 15
        self.MAT_Moto_spd = self.isg_mot_spd_filt  # 更新最终输出速度

        return self.MAT_Moto_spd


class VehicleTrajectoryFollower:
    def __init__(self, trajectory_csv):
        """
        初始化，读取轨迹点
        :param trajectory_csv: 包含轨迹点的CSV文件路径，轨迹点格式为[经度, 纬度, 航向角度]
        """
        self.dl = 1  # 轨迹点之间的间隔
        self.cx, self.cy, self.cyaw, self.ck = read_csv(trajectory_csv)
        self.sp = calc_speed_profile(self.cx, self.cy, self.cyaw, TARGET_SPEED)  # 计算速度轨迹
        self.init_mpc()

    def init_mpc(self):
        self.goal = [self.cx[-1], self.cy[-1]]
        self.state = State(x=self.cx[0], y=self.cy[0], yaw=self.cyaw[0], v=0.0)

        # 初始航向角补偿
        if self.state.yaw - self.cyaw[0] >= math.pi:
            self.state.yaw -= math.pi * 2.0
        elif self.state.yaw - self.cyaw[0] <= -math.pi:
            self.state.yaw += math.pi * 2.0

        self.target_ind, _ = self.calc_nearest_index(self.state, self.cx, self.cy, self.cyaw, 0)
        self.odelta, self.oa = None, None
        self.cyaw = self.smooth_yaw(self.cyaw)

        # 初始化转向角滤波参数
        self.previous_turn_angle = 0.0
        self.max_turn_rate = 4.0  # 根据需求调整

    def update_target_index(self, ego_state, ego_yaw, ego_v):
        self.state.x = ego_state[0]
        self.state.y = ego_state[1]
        self.state.yaw = ego_yaw
        self.state.v = ego_v
        self.state.predelta = 0
        self.target_ind, _ = self.calc_nearest_index(self.state, self.cx, self.cy, self.cyaw, self.target_ind)

    def calculate_turn_angle(self, ego_state, ego_yaw, ego_v):
        self.state.x = ego_state[0]
        self.state.y = ego_state[1]
        self.state.yaw = ego_yaw
        self.state.v = ego_v
        self.state.predelta = 0

        # 计算参考轨迹
        self.xref, self.target_ind, self.dref = self.calc_ref_trajectory(
            self.state, self.cx, self.cy, self.cyaw, self.ck, self.sp, self.dl, self.target_ind
        )

        # 当前状态
        self.x0 = [self.state.x, self.state.y, self.state.v, self.state.yaw]

        # 使用MPC控制器计算控制输入
        oa, odelta, ox, oy, oyaw, ov = self.iterative_linear_mpc_control(
            self.xref, self.x0, self.dref, self.oa, self.odelta
        )
        
        # plt.figure()
        # plt.plot(ox, oy, c='r')
        # plt.plot(self.cx, self.cy, c='g')
        # plt.plot(self.xref[0],self.xref[1],c='b')
        # plt.axis("equal")
        # plt.scatter(self.state.x, self.state.y, c='y')
        # plt.scatter(self.cx[self.target_ind], self.cy[self.target_ind], c='b')
        # plt.savefig("prediction_points.png")
        
        print(self.state.x,self.state.y,self.state.v,self.state.yaw)
        print(self.cx[self.target_ind], self.cy[self.target_ind],self.cyaw[self.target_ind])

        di, ai = 0.0, 0.0
        if odelta is not None:
            di = -odelta[0]
            ai = oa[0]
            logging.debug(f"Calculated steering angle (rad): {di}, acceleration: {ai}")

            # 限制转向角
            di = np.clip(di, -MAX_STEER, MAX_STEER)
            print(math.degrees(di))
            # 将转向角转换为度并平滑处理
            filtered_angle = self.smooth_turn_angle(math.degrees(di) * WHEEL_FACTOR)

            # 更新MPC控制器的上一轮控制输入
            self.oa = oa
            self.odelta = odelta
            return filtered_angle, ai
        else:
            logging.warning("MPC computation failed, using default values.")
            return 0, 0

    def calc_ref_trajectory(self, state, cx, cy, cyaw, ck, sp, dl, pind):
        xref = np.zeros((NX, T + 1))
        dref = np.zeros((1, T + 1))
        ncourse = len(cx)

        ind, _ = self.calc_nearest_index(state, cx, cy, cyaw, pind)

        if pind >= ind:
            ind = pind

        xref[0, 0] = cx[ind]
        xref[1, 0] = cy[ind]
        xref[2, 0] = sp[ind]
        xref[3, 0] = cyaw[ind]
        dref[0, 0] = 0.0  # steer operational point should be 0

        travel = 0.0

        for i in range(T + 1):
            travel += abs(state.v) * DT  # 累计形式的距离
            dind = int(round(travel / dl))  # dl是路径点的间隔，travel/dl是当前车辆已经行驶的路径点数

            if (ind + i) < ncourse:
                xref[0, i] = cx[ind + i]
                xref[1, i] = cy[ind + i]
                xref[2, i] = sp[ind + i]
                xref[3, i] = cyaw[ind + i]
                dref[0, i] = 0.0
            else:
                xref[0, i] = cx[ncourse - 1]
                xref[1, i] = cy[ncourse - 1]
                xref[2, i] = sp[ncourse - 1]
                xref[3, i] = cyaw[ncourse - 1]
                dref[0, i] = 0.0

        return xref, ind, dref

    def smooth_yaw(self, yaw):
        for i in range(len(yaw) - 1):
            dyaw = yaw[i + 1] - yaw[i]

            while dyaw >= math.pi / 2.0:
                yaw[i + 1] -= math.pi * 2.0
                dyaw = yaw[i + 1] - yaw[i]

            while dyaw <= -math.pi / 2.0:
                yaw[i + 1] += math.pi * 2.0
                dyaw = yaw[i + 1] - yaw[i]

        return yaw

    def calc_nearest_index(self, state, cx, cy, cyaw, pind):
        dx = [state.x - icx for icx in cx[pind:pind + N_IND_SEARCH]]
        dy = [state.y - icy for icy in cy[pind:pind + N_IND_SEARCH]]

        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

        mind = min(d)
        ind = d.index(mind) + pind
        mind = math.sqrt(mind)  # 最小距离

        dxl = cx[ind] - state.x
        dyl = cy[ind] - state.y

        angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
        if angle < 0:
            mind *= -1
        print("nearest indx: ",ind)
        return ind, mind
    
    def iterative_linear_mpc_control(self, xref, x0, dref, oa, od):
        ox, oy, oyaw, ov = None, None, None, None

        if oa is None or od is None:
            oa = [0.0] * T  # 上一轮优化得到的加速度序列，如果是第一次迭代，就初始化为0
            od = [0.0] * T  # 上一轮优化得到的转角序列，如果是第一次迭代，就初始化为0

        for i in range(MAX_ITER):
            xbar = self.predict_motion(x0, oa, od, xref)
            poa, pod = oa[:], od[:]
            oa, od, ox, oy, oyaw, ov = self.linear_mpc_control(xref, xbar, x0, dref)
            du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value 用于判断是否收敛
            if du <= DU_TH:  # 如果u的变化小于阈值，说明收敛了, 就退出迭代
                break
        else:
            logging.warning("Iterative is max iter")

        return oa, od, ox, oy, oyaw, ov

    def linear_mpc_control(self, xref, xbar, x0, dref):
        """
        linear mpc control

        xref: reference point
        xbar: operational point
        x0: initial state
        dref: reference steer angle
        """

        x = cvxpy.Variable((NX, T + 1))
        u = cvxpy.Variable((NU, T))

        cost = 0.0
        constraints = []

        for t in range(T):
            cost += cvxpy.quad_form(u[:, t], R)  # cost function 控制输入的cost

            if t != 0:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)  # state cost function

            A, B, C = get_linear_model_matrix(
                xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]  # 动力学约束

            if t < (T - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)  # input difference cost
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= MAX_DSTEER * DT]

        cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)  # final state cost

        constraints += [x[:, 0] == x0]
        constraints += [x[2, :] <= MAX_SPEED]  # 速度上限
        constraints += [x[2, :] >= MIN_SPEED]  # 速度下限
        constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]  # 加速度约束
        constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]  # 转向角约束

        # 使用更快的求解器 OSQP
        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.OSQP, verbose=False)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox = get_nparray_from_matrix(x.value[0, :])
            oy = get_nparray_from_matrix(x.value[1, :])
            ov = get_nparray_from_matrix(x.value[2, :])
            oyaw = get_nparray_from_matrix(x.value[3, :])
            oa = get_nparray_from_matrix(u.value[0, :])
            odelta = get_nparray_from_matrix(u.value[1, :])
        else:
            logging.error("Error: Cannot solve MPC.")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov

    def predict_motion(self, x0, oa, od, xref):
        xbar = xref * 0.0
        for i, _ in enumerate(x0):
            xbar[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for ai, di, i in zip(oa, od, range(1, T + 1)):
            state = self.update_state(state, ai, di)
            xbar[0, i] = state.x
            xbar[1, i] = state.y
            xbar[2, i] = state.v
            xbar[3, i] = state.yaw

        return xbar

    def update_state(self, state, a, delta):
        # 输入检查
        if delta >= MAX_STEER:
            delta = MAX_STEER
        elif delta <= -MAX_STEER:
            delta = -MAX_STEER

        state.x += state.v * math.cos(state.yaw) * DT
        state.y += state.v * math.sin(state.yaw) * DT
        state.yaw += state.v / WB * math.tan(delta) * DT
        state.v += a * DT

        # 速度限制
        state.v = np.clip(state.v, MIN_SPEED, MAX_SPEED)

        return state

    def smooth_turn_angle(self, turn_angle):
        # 限制转向角的最大变化速率
        angle_difference = turn_angle - self.previous_turn_angle
        if angle_difference > self.max_turn_rate:
            update_turn_angle = self.previous_turn_angle + self.max_turn_rate
        elif angle_difference < -self.max_turn_rate:
            update_turn_angle = self.previous_turn_angle - self.max_turn_rate
        else:
            update_turn_angle = turn_angle

        # 更新上一次的转向角
        self.previous_turn_angle = update_turn_angle
        return update_turn_angle


class Can_use:
    def __init__(self, zone, shared_data):
        self.bus_ins = can.interface.Bus(channel='can0', bustype='socketcan')
        self.bus_vcu = can.interface.Bus(channel='can1', bustype='socketcan')
        self.shared_data = shared_data
        self.running = True
        self.thread = threading.Thread(target=self.read_can_data)
        self.thread.start()

    def read_can_data(self):
        while self.running:
            try:
                message_ins = self.bus_ins.recv(timeout=0.01)
                if message_ins:
                    self.parse_ins_message(message_ins)

                message_vcu = self.bus_vcu.recv(timeout=0.01)
                if message_vcu:
                    self.parse_vcu_message(message_vcu)
            except can.CanError:
                logging.warning("CAN 总线读取错误")
                continue

    def parse_ins_message(self, message_ins):
        with self.shared_data['lock']:
            if message_ins.arbitration_id == 0x504:
                # 解析经纬度
                INS_Latitude = (message_ins.data[0] << 24) | (message_ins.data[1] << 16) | (message_ins.data[2] << 8) | message_ins.data[3]
                INS_Longitude = (message_ins.data[4] << 24) | (message_ins.data[5] << 16) | (message_ins.data[6] << 8) | message_ins.data[7]
                INS_Latitude = INS_Latitude * 0.0000001 - 180
                INS_Longitude = INS_Longitude * 0.0000001 - 180

                ego_x, ego_y = latlon_to_utm(INS_Longitude, INS_Latitude)
                self.shared_data['ego_x'] = ego_x
                self.shared_data['ego_y'] = ego_y
                logging.debug(f"Parsed Position - X: {ego_x}, Y: {ego_y}")

            elif message_ins.arbitration_id == 0x505:
                # 解析速度信息
                INS_NorthSpd = (message_ins.data[0] << 8) | message_ins.data[1]
                INS_NorthSpd = INS_NorthSpd * 0.0030517 - 100  # m/s
                INS_NorthSpd *= 3.6

                INS_EastSpd = (message_ins.data[2] << 8) | message_ins.data[3]
                INS_EastSpd = INS_EastSpd * 0.0030517 - 100  # m/s
                INS_EastSpd *= 3.6

                INS_ToGroundSpd = (message_ins.data[4] << 8) | message_ins.data[5]
                INS_ToGroundSpd = INS_ToGroundSpd * 0.0030517 - 100  # m/s
                INS_ToGroundSpd *= 3.6

                speed = sqrt(INS_EastSpd**2 + INS_NorthSpd**2 + INS_ToGroundSpd**2)
                self.shared_data['ego_v'] = speed
                logging.debug(f"Parsed Speed: {speed} m/s")

            elif message_ins.arbitration_id == 0x502:
                # 解析航向角
                HeadingAngle = (message_ins.data[4] << 8) | message_ins.data[5]
                HeadingAngle = HeadingAngle * 0.010986 - 360
                utm_yaw_deg = 90 - HeadingAngle
                utm_yaw_rad = math.radians(utm_yaw_deg)

                # 平滑航向角
                previous_yaw = self.shared_data.get('previous_yaw', 0.0)
                smoothed_yaw = smooth_yaw_iter(previous_yaw, utm_yaw_rad)
                self.shared_data['previous_yaw'] = smoothed_yaw
                self.shared_data['ego_yaw'] = smoothed_yaw
                self.shared_data['ego_yaw_deg'] = math.degrees(smoothed_yaw)
                logging.debug(f"Parsed Yaw: {self.shared_data['ego_yaw_deg']} degrees")

            elif message_ins.arbitration_id == 0x500:
                # 解析加速度
                ACC_X = (message_ins.data[0] << 8) | message_ins.data[1]
                ACC_X = (ACC_X * 0.0001220703125 - 4) * 9.8  # 转换为 m/s²
                self.shared_data['ego_a'] = ACC_X
                logging.debug(f"Parsed Acceleration: {ACC_X} m/s²")

    def parse_vcu_message(self, message_vcu):
        with self.shared_data['lock']:
            if message_vcu.arbitration_id == 0x15C:
                allow_value = message_vcu.data[2] & 0x01
                self.shared_data['auto_driver_allowed'] = (allow_value == 1)
                logging.debug(f"Auto Driver Allowed: {self.shared_data['auto_driver_allowed']}")

            elif message_vcu.arbitration_id == 0x124:
                self.shared_data['eps_mode'] = (message_vcu.data[6] >> 4) & 0x03
                logging.debug(f"EPS Mode: {self.shared_data['eps_mode']}")

    def publish_planner_action(self, action, id, action_type, mod, enable):
        """将规划动作发布到CAN"""
        if action_type == "angle":
            # 数据缩放和转换
            data1 = int((action - (-738)) / 0.1)  # 根据需求调整
            data1_high = (data1 >> 8) & 0xFF
            data1_low = data1 & 0xFF

            data2 = int(mod) & 0x03
            data3 = int(250 / 10) & 0xFF  # 假设 angle_spd=25
            data4 = int(enable) & 0x01

            data = [data1_high, data1_low, data2, data3, data4, 0, 0, 0]

            msg = can.Message(arbitration_id=id, data=data, is_extended_id=False)
            try:
                self.bus_vcu.send(msg)
                logging.debug(f"Sent angle action: {data}")
            except can.CanError:
                logging.error("发送 CAN 消息失败")

        elif action_type == "acc":
            auto_drive_cmd_bits = mod & 0x07
            desired_speed = action[0]
            speed_scaled = int(desired_speed) & 0x1F
            byte0 = (speed_scaled << 3) | auto_drive_cmd_bits

            angle_scaled = int((action[1] - (-500)) / 0.1) & 0xFFFF
            byte1 = (angle_scaled >> 8) & 0xFF
            byte2 = angle_scaled & 0xFF

            acc = action[2]
            acc_scaled = int((acc - (-4)) / 1) & 0xFF

            data_666 = [byte0, byte1, byte2, acc_scaled, 0, 0, 0, 0]

            msg = can.Message(arbitration_id=id, data=data_666, is_extended_id=False)
            try:
                self.bus_vcu.send(msg)
                logging.debug(f"Sent acc action: {data_666}")
            except can.CanError:
                logging.error("发送 CAN 消息失败")

    def stop(self):
        self.running = False
        self.thread.join()


def on_press(key):
    global manual_triggered
    global stop_record
    try:
        if key.char == 's':
            manual_triggered = True
            logging.info("收到键盘输入's'，手动请求自动驾驶模式")
        if key.char == 'q':
            manual_triggered = False
            logging.info("收到键盘输入'q'，退出自动驾驶模式")
        if key.char == "x":
            stop_record = True
            logging.info("收到键盘输入'x'，停止记录")
    except AttributeError:
        if key == keyboard.Key.esc:
            logging.info("收到Esc键，退出程序")
            return False  # 停止监听


def keyboard_listener():
    # 创建并启动键盘监听器线程
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def start_main(shared_data, can_use, follower, filter_obj):
    global mod_666
    global mod_AE
    global manual_triggered
    
        
    with shared_data['lock']:
        follower.update_target_index((shared_data['ego_x'], shared_data['ego_y']), shared_data['ego_yaw'], shared_data['ego_v'])

    while True:
        with shared_data['lock']:
            eps_mode = shared_data.get('eps_mode', 2)
            if manual_triggered and eps_mode != 3:
                mod_AE = 1
                mod_666 = 1
            if eps_mode == 3:
                logging.info("==========================reset============================")
                mod_AE = 3
                mod_666 = 0
                manual_triggered = False

            if mod_AE == 1 and mod_666 == 1:
                ego_x = shared_data.get('ego_x', None)
                ego_y = shared_data.get('ego_y', None)
                ego_yaw = shared_data.get('ego_yaw', 0.0)
                ego_v = shared_data.get('ego_v', 0.0)

        if mod_AE == 1 and mod_666 == 1:
            if ego_x is not None and ego_y is not None:
                with shared_data['lock']:
                    follower.update_target_index((shared_data['ego_x'], shared_data['ego_y']), shared_data['ego_yaw'], shared_data['ego_v'])

                turn_angle, ai = follower.calculate_turn_angle(
                    (ego_x, ego_y, ego_yaw),
                    ego_yaw,
                    ego_v
                )
                filtered_angle = filter_obj.update_speed(turn_angle)
                print(filtered_angle)
                new_frame = [5, filtered_angle, 0]
            else:
                logging.warning("主车定位丢失...")
                new_frame = [0, 0, 0]
        else:
            logging.info("请按's'进入自动驾驶模式...")
            new_frame = [0, 0, 0]

        with shared_data['lock']:
            shared_data['frame'] = new_frame

        time.sleep(0.01)  # 每次循环后休眠10ms


def send_frame(shared_data, can_use):
    while True:
        time.sleep(0.005)  # 每隔5ms发送一帧（200帧每秒）
        with shared_data["lock"]:
            if shared_data["frame"] is not None:
                last_frame = shared_data["frame"]
            else:
                last_frame = [0, 0, 0]

        if last_frame is not None:
            can_use.publish_planner_action(action=last_frame, id=0x666, action_type="acc", mod=1, enable=1)


def main():
    # 使用示例
    trajectory_csv = '/home/nvidia/vcii/follow_trajectory/collect_trajectory/processed_straight12_17_with_yaw_ck.csv'
    follower = VehicleTrajectoryFollower(trajectory_csv)
    filter_obj = ISGSpeedFilter()

    # 用于在线程之间共享数据
    shared_data = {
        "frame": None,  # 存储最新的帧
        "lock": threading.Lock(),  # 用于保证线程安全的锁
        "ego_x": None,
        "ego_y": None,
        "ego_yaw": 0.0,
        "ego_v": 0.0,
        "ego_a": 0.0,
        "eps_mode": 2,
        "auto_driver_allowed": False,
        "previous_yaw": 0.0
    }

    # 初始化canbus
    can_use = Can_use(zone=50, shared_data=shared_data)

    # 创建计算线程和发送线程
    compute_thread = threading.Thread(target=start_main, args=(shared_data, can_use, follower, filter_obj))
    send_thread = threading.Thread(target=send_frame, args=(shared_data, can_use))
    # 键盘监听线程
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)

    # 启动线程
    keyboard_thread.start()
    compute_thread.start()
    send_thread.start()

    try:
        # 主线程等待计算和发送线程完成（通常不会退出）
        compute_thread.join()
        send_thread.join()
    except KeyboardInterrupt:
        logging.info("程序被中断")
    finally:
        can_use.stop()


if __name__ == '__main__':
    main()
