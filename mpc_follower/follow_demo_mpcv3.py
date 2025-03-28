"如果转角阈值过大或者过小，做一个滤波"

from pyproj import Proj, Transformer,transform,CRS
import sys
from math import radians, cos, sin, asin, sqrt, degrees, atan2
import matplotlib.pyplot as plt
import os 
import can
from math import radians, cos, sin, asin, sqrt, degrees, atan2
import matplotlib.pyplot as plt
import time
import numpy as np
import threading
import math
from read_csv import read_csv
from pynput import keyboard
import cvxpy
import csv



# mpc parameters
NX = 3  # x = x, y, yaw
NU = 2  # u = [v,delta]
T = 5 # horizon length
R = np.diag([0.1, 0.1])  # input cost matrix
Rd = np.diag([0.1, 0.1])  # input difference cost matrix
Q = np.diag([10, 10, 2])  # state cost matrix
Qf = Q  # state final matrix


#车辆
dt = 0.33  # 时间间隔，单位：s
L = 3 # 车辆轴距，单位：m
v = 2.7  # 初始速度
x_0 = 0  # 初始x
y_0 = 0  # 初始y
psi_0 = 90  # 初始航向角

MAX_STEER = np.deg2rad(55.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(55.0)  # maximum steering speed [rad/s]

MAX_VEL = 10.0  # maximum accel [m/s]


# 车辆参数
VEHICLE_WIDTH = 1.9   # m
VEHICLE_LENGTH = 4.5  # m
WHEEL_FACTOR = 7.2
manual_triggered = False
stop_record = False
mod_666 = 0
mod_AE = 0


def get_nparray_from_matrix(x):
    return np.array(x).flatten()




# 初始化 WGS84 到 UTM 的投影转换器
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

    :param previous_yaw: (float) Previous yaw angle in radians
    :param new_yaw: (float) New yaw angle in radians (not yet normalized)
    :return: (float) Smoothed and normalized yaw angle in radians within (-pi, pi]
    """
    dyaw = new_yaw - previous_yaw

    # 调整 dyaw，使其在 [-pi, pi] 范围内
    dyaw = (dyaw + np.pi) % (2.0 * np.pi) - np.pi

    # 平滑后的 yaw
    smoothed_yaw = previous_yaw + dyaw

    return smoothed_yaw

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


def smooth_yaw(yaw):
    print("yaw",yaw)
    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw

class KinematicModel_3:
  """假设控制量为转向角delta_f和加速度a
  """

  def __init__(self, x, y, psi, v, L, dt):
    self.x = x
    self.y = y
    self.psi = psi
    self.v = v
    self.L = L
    # 实现是离散的模型
    self.dt = dt

  def update_state(self, a, delta_f):
    print("delta_f",delta_f)
    print("self.psi",self.psi)
    self.x = self.x+self.v*math.cos(self.psi)*self.dt
    self.y = self.y+self.v*math.sin(self.psi)*self.dt
    self.psi = self.psi+self.v/self.L*math.tan(delta_f)*self.dt
    # self.psi = normalize_angle(self.psi)
    self.v = self.v+a*self.dt

  def get_state(self):
    return self.x, self.y, self.psi, self.v

  def state_space(self, ref_delta, ref_yaw):
    """将模型离散化后的状态空间表达

    Args:
        ref_delta (_type_): 参考的转角控制量
        ref_yaw (_type_): 参考的偏航角

    Returns:
        _type_: _description_
    """
    # print(ref_delta)
    A = np.matrix([
        [1.0, 0.0, -self.v*self.dt*math.sin(ref_yaw)],
        [0.0, 1.0, self.v*self.dt*math.cos(ref_yaw)],
        [0.0, 0.0, 1.0]])

    B = np.matrix([
        [self.dt*math.cos(ref_yaw), 0],
        [self.dt*math.sin(ref_yaw), 0],
        [self.dt*math.tan(ref_delta)/self.L, self.v*self.dt /(self.L*math.cos(ref_delta)*math.cos(ref_delta))]
    ])

    C = np.eye(3)
    return A, B, C


class VehicleTrajectoryFollower:
    def __init__(self, trajectory_csv):
        """
        初始化，读取轨迹点
        :param trajectory_csv: 包含轨迹点的CSV文件路径，轨迹点格式为[经度, 纬度, 航向角度]
        """
        self.dl = 1  # 轨迹点之间的间隔
        self.reference_path = MyReferencePath(trajectory_csv)
        self.goal = self.reference_path.refer_path[-1, 0:2]
        self.x_0 = self.reference_path.refer_path[0, 0]
        self.y_0 = self.reference_path.refer_path[0, 1]
        self.psi_0 = self.reference_path.refer_path[0, 2]
        self.ugv = KinematicModel_3(self.x_0, self.y_0, self.psi_0, v, L, dt)
        
        self.previous_turn_angle = 0
        self.max_turn_rate = 6
        
    def calculate_turn_angle(self, ego_state, ego_yaw, ego_v):
        
        self.ugv.x = ego_state[0]
        self.ugv.y = ego_state[1]
        self.ugv.psi = ego_yaw
        self.ugv.v = 2.7
        
        
        robot_state = np.zeros(4)
        robot_state[0] = ego_state[0]
        robot_state[1] = ego_state[1]
        robot_state[2] = ego_yaw
        robot_state[3] = ego_v
        x0 = robot_state[0:3]
        
        xref, ind, dref = self.reference_path.calc_ref_trajectory(robot_state, self.dl)
        opt_v, opt_delta, opt_x, opt_y, opt_yaw = linear_mpc_control(xref, x0, dref, self.ugv)
        di = opt_delta[0]

        if di >= MAX_STEER:
            di = MAX_STEER
        elif di <= -MAX_STEER:
            di = -MAX_STEER

        plt.figure()
        plt.plot(opt_x, opt_y, c='r')
        plt.plot(self.reference_path.refer_path[:, 0], self.reference_path.refer_path[:, 1], c='g')
        plt.plot(xref[0],xref[1],c='b')
        plt.axis("equal")
        plt.scatter(self.ugv.x, self.ugv.y, c='y')
        plt.scatter(self.reference_path.refer_path[ind][0], self.reference_path.refer_path[ind][1], c='b')
        plt.savefig("prediction_points.png")

        print(xref)
        print(opt_x,opt_y,opt_yaw)
        # -pi到pi之间
        # result_angle = normalize_angle(result_angle)
        # 转为角度
        print("di rad: ", di)



        result_angle = math.degrees(di)
        # 平滑转向角
        print("result angle: ", result_angle)
        final_angle = self.smooth_turn_angle(-result_angle*7.2)
        print("final angle: ", final_angle)
        return final_angle

    # 用于平滑转角
    def smooth_turn_angle(self, turn_angle):
        # 限制转向角的最大变化速率
        angle_difference = turn_angle - self.previous_turn_angle
        if angle_difference > self.max_turn_rate:
            update_turn_angle = self.previous_turn_angle + 4
        elif angle_difference < -self.max_turn_rate:
            update_turn_angle = self.previous_turn_angle - 4
        else:
            update_turn_angle = turn_angle

        # 更新上一次的转向角
        self.previous_turn_angle = update_turn_angle
        return update_turn_angle

class MyReferencePath:
    def __init__(self, trajectory_csv):
        self.refer_path = np.array(self.read_csv(trajectory_csv))
        print(self.refer_path)
        for i in range(len(self.refer_path)):
            if i == 0:
                dx = self.refer_path[i+1, 0] - self.refer_path[i, 0]
                dy = self.refer_path[i+1, 1] - self.refer_path[i, 1]
                ddx = self.refer_path[2, 0] + \
                    self.refer_path[0, 0] - 2*self.refer_path[1, 0]
                ddy = self.refer_path[2, 1] + \
                    self.refer_path[0, 1] - 2*self.refer_path[1, 1]
            elif i == (len(self.refer_path)-1):
                dx = self.refer_path[i, 0] - self.refer_path[i-1, 0]
                dy = self.refer_path[i, 1] - self.refer_path[i-1, 1]
                ddx = self.refer_path[i, 0] + \
                    self.refer_path[i-2, 0] - 2*self.refer_path[i-1, 0]
                ddy = self.refer_path[i, 1] + \
                    self.refer_path[i-2, 1] - 2*self.refer_path[i-1, 1]
            else:
                dx = self.refer_path[i+1, 0] - self.refer_path[i, 0]
                dy = self.refer_path[i+1, 1] - self.refer_path[i, 1]
                ddx = self.refer_path[i+1, 0] + \
                    self.refer_path[i-1, 0] - 2*self.refer_path[i, 0]
                ddy = self.refer_path[i+1, 1] + \
                    self.refer_path[i-1, 1] - 2*self.refer_path[i, 1]
            self.refer_path[i, 2] = math.atan2(dy, dx)  # yaw
            # 计算曲率:设曲线r(t) =(x(t),y(t)),则曲率k=(x'y" - x"y')/((x')^2 + (y')^2)^(3/2).
            # 参考：https://blog.csdn.net/weixin_46627433/article/details/123403726
            self.refer_path[i, 3] = (
                ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))  # 曲率k计算
        print(self.refer_path)
        self.refer_path[:,2] = smooth_yaw(self.refer_path[:, 2])
         
    def read_csv(self, csv_file_path): 
        trajecotry_data = []

        # 打开CSV文件并读取内容  
        with open(csv_file_path, mode='r', newline='') as file:  
            csv_reader = csv.reader(file)  
            
            # 跳过标题行（如果有的话）  
            headers = next(csv_reader, None)  # 这行代码会读取第一行，如果第一行是标题则跳过  
            
            # 读取每一行数据并添加到列表中  
            for row in csv_reader:  
                lon = float(row[0])  
                lat = float(row[1])
                heading = float(row[2])
                curve = float(row[3])
                # 将经纬度转换为UTM坐标
                trajecotry_data.append([lon, lat, heading, curve])
                # 将UTM坐标和航向角存储到traj_data中  
        return trajecotry_data

    def calc_track_error(self, x, y):
        """计算跟踪误差

        Args:
            x (_type_): 当前车辆的位置x
            y (_type_): 当前车辆的位置y

        Returns:
            _type_: _description_
        """
        # 寻找参考轨迹最近目标点
        d_x = [self.refer_path[i, 0]-x for i in range(len(self.refer_path))]
        d_y = [self.refer_path[i, 1]-y for i in range(len(self.refer_path))]
        d = [np.sqrt(d_x[i]**2+d_y[i]**2) for i in range(len(d_x))]
        s = np.argmin(d)  # 最近目标点索引
        print("nearest idx: ", s)
        yaw = self.refer_path[s, 2]
        k = self.refer_path[s, 3]
        angle = normalize_angle(yaw - math.atan2(d_y[s], d_x[s]))
        e = d[s]  # 误差
        if angle < 0:
            e *= -1

        return e, k, yaw, s

    def calc_ref_trajectory(self, robot_state, dl=1.0):
        """计算参考轨迹点，统一化变量数组，便于后面MPC优化使用
            参考自https://github.com/AtsushiSakai/PythonRobotics/blob/eb6d1cbe6fc90c7be9210bf153b3a04f177cc138/PathTracking/model_predictive_speed_and_steer_control/model_predictive_speed_and_steer_control.py
        Args:
            robot_state (_type_): 车辆的状态(x,y,yaw,v)
            dl (float, optional): _description_. Defaults to 1.0.

        Returns:
            _type_: _description_
        """
        # 寻找参考轨迹最近目的点，计算与最近点的误差，最近点的k曲率，最近点的yaw参考点航向角，最近点的索引
        e, k, ref_yaw, temp_ind = self.calc_track_error(
            robot_state[0], robot_state[1])
        ind = temp_ind + 5

        xref = np.zeros((NX, T + 1))
        dref = np.zeros((NU, T))
        ncourse = len(self.refer_path)

        # 状态量
        xref[0, 0] = self.refer_path[ind, 0]
        xref[1, 0] = self.refer_path[ind, 1]
        xref[2, 0] = self.refer_path[ind, 2]

        # 参考控制量[v,delta]
        ref_delta = math.atan2(L*k, 1)
        dref[0, :] = robot_state[3]
        dref[1, :] = ref_delta
        
        # for i in range(NU):
        #     temp_k = self.refer_path[ind+0][3]
        #     temp_ref_delta = math.atan2(L*temp_k,1)
        #     dref[1, i] = temp_ref_delta
        # print(dref)

        travel = 0.0

        for i in range(T + 1):
            travel += abs(robot_state[3]) * dt
            dind = int(round(travel / dl))

            if (ind + dind) < ncourse:
                # xref[0, i] = self.refer_path[ind + dind, 0]
                # xref[1, i] = self.refer_path[ind + dind, 1]
                # xref[2, i] = self.refer_path[ind + dind, 2]
                xref[0, i] = self.refer_path[ind + i, 0]
                xref[1, i] = self.refer_path[ind + i, 1]
                xref[2, i] = self.refer_path[ind + i, 2]

            else:
                xref[0, i] = self.refer_path[ncourse - 1, 0]
                xref[1, i] = self.refer_path[ncourse - 1, 1]
                xref[2, i] = self.refer_path[ncourse - 1, 2]
        # print("xref",xref)
        # print("dref",dref)
        return xref, ind, dref




    # 用于平滑转角
    def smooth_turn_angle(self, turn_angle):
        # 限制转向角的最大变化速率
        angle_difference = turn_angle - self.previous_turn_angle
        if angle_difference > self.max_turn_rate:
            update_turn_angle = self.previous_turn_angle + 4
        elif angle_difference < -self.max_turn_rate:
            update_turn_angle = self.previous_turn_angle - 4
        else:
            update_turn_angle = turn_angle

        # 更新上一次的转向角
        self.previous_turn_angle = update_turn_angle
        return update_turn_angle


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].

    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    copied from https://atsushisakai.github.io/PythonRobotics/modules/path_tracking/stanley_control/stanley_control.html
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def linear_mpc_control(xref, x0, delta_ref, ugv):
    """
    linear mpc control

    xref: reference point
    x0: initial state
    delta_ref: reference steer angle
    ugv:车辆对象
    returns: 最优的控制量和最优状态
    """

    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0  # 代价函数
    constraints = []  # 约束条件

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t]-delta_ref[:, t], R)

        if t != 0:
            cost += cvxpy.quad_form(x[:, t] - xref[:, t], Q)

        A, B, C = ugv.state_space(delta_ref[1, t], xref[2, t])
        constraints += [x[:, t + 1]-xref[:, t+1] == A @
                        (x[:, t]-xref[:, t]) + B @ (u[:, t]-delta_ref[:, t])]


    cost += cvxpy.quad_form(x[:, T] - xref[:, T], Qf)

    constraints += [(x[:, 0]) == x0]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_VEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        opt_x = get_nparray_from_matrix(x.value[0, :])
        opt_y = get_nparray_from_matrix(x.value[1, :])
        opt_yaw = get_nparray_from_matrix(x.value[2, :])
        opt_v = get_nparray_from_matrix(u.value[0, :])
        opt_delta = get_nparray_from_matrix(u.value[1, :])

    else:
        print("Error: Cannot solve mpc..")
        opt_v, opt_delta, opt_x, opt_y, opt_yaw = None, None, None, None, None,

    
    return opt_v, opt_delta, opt_x, opt_y, opt_yaw


class Can_use:
    def __init__(self, zone):
        self.bus_ins = can.interface.Bus(channel='can0', bustype='socketcan')
        self.bus_vcu = can.interface.Bus(channel='can1', bustype='socketcan')
        self.ego_lon = 31.8925019
        self.ego_lat = 118.8171577
        self.ego_yaw_deg =  90
        self.ego_yaw = math.radians(self.ego_yaw_deg)
        self.ego_v =  3
        self.ego_a = 0
        self.eps_mode = 2
        
        self.ego_x = 0
        self.ego_y = 0
        
        # 用于平滑航向角
        self.previous_yaw = math.radians(self.ego_yaw_deg)  # 初始 yaw 转换为弧度

        
    def read_ins_info(self):
        """获取惯导的主车信息"""
        message_ins = self.bus_ins.recv()
        message_vcu = self.bus_vcu.recv()

        # print(message_ins.arbitration_id == 0x504)
        if message_ins is not None and message_ins.arbitration_id == 0x504:
            # 直接获取数据字节
            can_data = message_ins.data
            # 解析前4个字节为纬度
            INS_Latitude = (can_data[0] << 24) | (can_data[1] << 16) | (can_data[2] << 8) | can_data[3]
            # 解析后4个字节为经度
            INS_Longitude = (can_data[4] << 24) | (can_data[5] << 16) | (can_data[6] << 8) | can_data[7]
            # INS_Latitude = (can_data[0] << 24) | (can_data[1] << 16) | (can_data[2] << 8) | can_data[3]
            INS_Latitude = INS_Latitude*0.0000001-180                   # 解析后4个字节为经度
            # INS_Longitude = (can_data[4] << 24) | (can_data[5] << 16) | (can_data[6] << 8) | can_data[7]
            INS_Longitude= INS_Longitude*0.0000001-180 
            # 将经纬度转换为 UTM 坐标
            ego_x, ego_y = latlon_to_utm(INS_Longitude, INS_Latitude)
            self.ego_x = ego_x
            self.ego_y = ego_y

             
        if message_ins is not None and message_ins.arbitration_id == 0x505:
            speed_data = message_ins.data
                    
            # 北向速度
            INS_NorthSpd =  (speed_data[0] << 8) | speed_data[1]
            INS_NorthSpd =   INS_NorthSpd*0.0030517-100    # m/s
            INS_NorthSpd *= 3.6
            # 东向速度
            INS_EastSpd =  (speed_data[2] << 8) | speed_data[3]
            INS_EastSpd =   INS_EastSpd*0.0030517-100    # m/s
            INS_EastSpd *= 3.6
            # 地向速度
            INS_ToGroundSpd =  (speed_data[4] << 8) | speed_data[5]
            INS_ToGroundSpd =   INS_ToGroundSpd*0.0030517-100    # m/s
            INS_ToGroundSpd *= 3.6
                    
            speed =  sqrt(INS_EastSpd**2+INS_NorthSpd**2+INS_ToGroundSpd**2)
                    
            # 计算航向角（单位：度）
            # angle = degrees(atan2(INS_NorthSpd, INS_EastSpd))
            self.ego_v = speed
        # else:
        #     self.ego_v = None

        if message_ins is not None and message_ins.arbitration_id == 0x502:
            # self.ego_yaw = angle
            Angle_data = message_ins.data
            HeadingAngle =  (Angle_data[4] << 8) | Angle_data[5]
            # HeadingAngle =   -(HeadingAngle*0.010986-360-90)
            HeadingAngle =   HeadingAngle*0.010986-360
            # self.ego_yaw = HeadingAngle
            
            
            # 将航向角从 INS 坐标系转换为 UTM 坐标系
            # INS: 0° 正北，东为正
            # UTM: 0° 正东，北为正
            # 转换公式：UTM_yaw = 90 - INS_yaw
            utm_yaw_deg = 90 - HeadingAngle
            print("utm yaw deg: ",utm_yaw_deg)
            utm_yaw_deg = normalize_angle(utm_yaw_deg)               
            
            utm_yaw_rad = math.radians(utm_yaw_deg)
            
            # 平滑航向角
            # smoothed_yaw = smooth_yaw_iter(self.previous_yaw, utm_yaw_rad)
            smoothed_yaw = utm_yaw_rad
            self.previous_yaw = smoothed_yaw
            self.ego_yaw = smoothed_yaw
            self.ego_yaw_deg = math.degrees(smoothed_yaw)  # 转换回度数用于其他部分
            print(" ego yaw in ins: ", self.ego_yaw)
 
        
 
            
        if message_ins is not None and message_ins.arbitration_id == 0x500:
            acc_data = message_ins.data
            # 北向速度
            ACC_X =  (acc_data[0] << 8) | acc_data[1]
            ACC_X =   (ACC_X*0.0001220703125-4)*9.8   # g
            self.ego_a = ACC_X
        
        if message_vcu is not None and message_vcu.arbitration_id == 0x15C:
            allow_value = message_vcu.data[2] & 0x01
            self.auto_driver_allowed = (allow_value == 1)

        if message_vcu is not None and message_vcu.arbitration_id == 0x124:
            eps_mode = (message_vcu.data[6] >> 4) & 0x03
            # print("========================",eps_mode)
            self.eps_mode = eps_mode

    def publish_planner_ation(self, action, id, action_type, mod, enable):
        """将规划动作发布到CAN"""
        # 验证输入参数类型和范围
        # if not isinstance(angle, (int, float)):
        #     print(f"Invalid angle value: {angle}")
        #     return
        if action_type == "angle":    
            # 数据缩放和转换
            # action = 0
            data1 = int((action - (-738)) / 0.1)  # 确保data1根据传入angle正确计算
            data1_high = (data1 >> 8) & 0xFF    # data1的高8位
            data1_low = data1 & 0xFF            # data1的低8位

            data2 = int(mod) & 0x03             # data2缩放到2位范围，0-3
            data3 = int(250 / 10) & 0xFF     # data3缩放到8位范围，0-255, angle_spd=100
            data4 = int(enable) & 0x01          # data4缩放到1位范围，0或1
                
            # 打印调试信息，检查缩放和转换过程
            # print(f"Original angle: {angle}, Scaled data1: {data1}, data1_high: {data1_high}, data1_low: {data1_low}")
            # print(f"Original data2: {data2}, data3: {data3}, data4: {data4}")

            # 构建发送数据，确保8字节长度
            data = [data1_high, data1_low, data2, data3, data4, 0, 0, 0]

            # 创建CAN消息，ID设置为0x0AE
            # print("angle id", id)
            msg = can.Message(arbitration_id=id, data=data, is_extended_id=False)
            self.bus_vcu.send(msg)
            # time.sleep(0.01)
        
        if action_type == "acc":
            auto_drive_cmd_bits = mod & 0x07  # 取最低3位
            # Auto speed cmd（位3-7）
            # 首先对速度进行缩放和偏移
            # 期望速度 单位km/h
            desired_speed = action[0] 
            speed_scaled = int(desired_speed) & 0x1F  # 取5位（位3-7）
            # 组合BYTE0
            byte0 = (speed_scaled << 3) | auto_drive_cmd_bits

            # BYTE1-BYTE2（需求方向盘转角）
            # 需要根据具体缩放因子和偏移量进行计算，假设缩放因子为0.1，偏移量为0
            angle_scaled = int((action[1] - (-500)) / 0.1) & 0xFFFF  # 16位
            byte1 = (angle_scaled >> 8) & 0xFF  # 高8位
            byte2 = angle_scaled & 0xFF         # 低8位

            # BYTE3（需求制动减速度）
            # 进行缩放和偏移
            acc  =  action[2]
            acc_scaled = int((acc - (-4)) / 1) & 0xFF  # 假设缩放因子1，偏移量-4

            # 构建发送数据，剩余字节填充0
            data_666 = [byte0, byte1, byte2, acc_scaled, 0, 0, 0, 0]
            print("data_666: ", data_666)
            msg = can.Message(arbitration_id=id, data=data_666, is_extended_id=False)
            # 发送CAN消息
            self.bus_vcu.send(msg)
            # time.sleep(0.01)

        # 限制发送频率
        # time.sleep(0.01)


def on_press(key):
    global manual_triggered
    global stop_record
    try:
        if key.char == 's':
            manual_triggered = True
            print("收到键盘输入's'，手动请求自动驾驶模式")
        if key.char == 'q':
            manual_triggered = False
        if key.char == "x":
            stop_record = True
    except AttributeError:
        if key == keyboard.Key.esc:
            print("收到Esc键，退出程序")
            return False  # 停止监听


def keyboard_listener():  
    # 创建并启动键盘监听器线程  
    with keyboard.Listener(on_press=on_press) as listener:  
        listener.join() 



def start_main(shared_data,can_use,follower,filter):
    global mod_666
    global mod_AE
    global manual_triggered

    for i in range(5):
        can_use.read_ins_info()

    # update_target_index
    # follower.update_target_index((can_use.ego_x, can_use.ego_y), can_use.ego_yaw, can_use.ego_v)
    
    while True:
        t0 = time.time()
        for i in range(5):
            can_use.read_ins_info()
            # print("can_use.ego_yaw: ",can_use.ego_yaw)

        if can_use.eps_mode != 3 and manual_triggered:
            mod_AE = 1
            mod_666 = 1
        if can_use.eps_mode == 3:
            # print("==========================reset============================")
            mod_AE = 3
            mod_666 = 0
            manual_triggered = False
        if mod_AE == 1 and mod_666 == 1:

            if can_use.ego_x is not None and can_use.ego_y is not None:     
                turn_angle = follower.calculate_turn_angle((can_use.ego_x, can_use.ego_y, can_use.ego_yaw), can_use.ego_yaw, can_use.ego_v)
                # turn_angle = 0
                # follower.update_target_index((can_use.ego_x, can_use.ego_y), can_use.ego_yaw, can_use.ego_v)
                # print("turn_angle=", turn_angle)
                filtered_angle = filter.update_speed(turn_angle)
                print("turn angle: ",turn_angle)
                print("time cost: ",time.time()-t0)
                new_frame = [5, filtered_angle, 0]     
            else:
                print("主车定位丢失...")
                new_frame = [0, 0, 0]
        else:
            print("请按s进入自动驾驶模式...")
            new_frame = [0, 0, 0]
            continue
        with shared_data['lock']:
            shared_data['frame'] = new_frame


def send_frame(shared_data,can_use):
    last_frame = None
    global mod_AE
    # print("here===================================")
    while True:
        # 每隔0.01秒发送一帧（100帧每秒）
        time.sleep(0.005)
        # 使用锁来读取共享数据
        with shared_data["lock"]:
            if shared_data["frame"] is not None:
                last_frame = shared_data["frame"]

        if last_frame != None:
            print("last frame: ", last_frame)
            can_use.publish_planner_ation(action = last_frame, id=0x666, action_type="acc", mod=1, enable=1)


def main():
    # 使用示例
    trajectory_csv = '/home/nvidia/vcii/follow_trajectory/collect_trajectory/processed_bigcycle1219_with_yaw_ck.csv'
    # 10
    follower = VehicleTrajectoryFollower(trajectory_csv)
    # 创建过滤器实例
    filter = ISGSpeedFilter()
    # 初始化canbus
    can_use = Can_use(zone=50)
 
    
    # 用于在线程之间共享数据
    shared_data = {
        "frame": None,  # 存储最新的帧
        "lock": threading.Lock()  # 用于保证线程安全的锁
    }
    
    # 创建计算线程和发送线程
    compute_thread = threading.Thread(target=start_main, args=(shared_data,can_use,follower,filter))
    send_thread = threading.Thread(target=send_frame, args=(shared_data,can_use))
    # 键盘监听线程
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)

    # 启动线程
    keyboard_thread.start()
    compute_thread.start()
    send_thread.start()
        
    # 主线程等待计算和发送线程完成（通常不会退出）
    compute_thread.join()
    send_thread.join()
        
        
    
if __name__ == '__main__':
    main()

