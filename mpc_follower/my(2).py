"""

Path tracking simulation with iterative linear model predictive control for speed and steer control

author: Atsushi Sakai (@Atsushi_twi)

"""
import matplotlib.pyplot as plt
import time
import cvxpy
import math
import numpy as np
import sys
import pathlib


import csv
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from utils.angle import angle_mod
from mpc_follower.follow_demo_mpcv3 import *


NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
T = 10  # horizon length

# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 1.0])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.5, 0.5])  # state cost matrix
Qf = Q  # state final matrix
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 500.0  # max simulation time

# iterative paramter
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
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)
    # print("最近点的索引：", ind, pind)
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

        # if (ind + dind) < ncourse:  #n course是路径点的总数，判断是否超出路径点的总数
            # xref[0, i] = cx[ind + dind]
            # xref[1, i] = cy[ind + dind]
            # xref[2, i] = sp[ind + dind]
            # xref[3, i] = cyaw[ind + dind]
            # dref[0, i] = 0.0
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


def check_goal(state, goal, tind, nind):

    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)

    isgoal = (d <= GOAL_DIS)

    if abs(tind - nind) >= 5:
        isgoal = False

    isstop = (abs(state.v) <= STOP_SPEED)

    if isgoal and isstop:
        return True

    return False


def do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state):
    """
    Simulation

    cx: course x position list
    cy: course y position list
    cy: course yaw position list
    ck: course curvature list
    sp: speed profile
    dl: course tick [m]

    """

    goal = [cx[-1], cy[-1]]

    state = initial_state

    # initial yaw compensation
    if state.yaw - cyaw[0] >= math.pi:
        state.yaw -= math.pi * 2.0
    elif state.yaw - cyaw[0] <= -math.pi:
        state.yaw += math.pi * 2.0
    print("state.yaw:", state.yaw)
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    d = [0.0]
    a = [0.0]
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)
    # print(target_ind)
    odelta, oa = None, None

    # cyaw =[ normalize_angle(yaw) for yaw in cyaw]
    cyaw = smooth_yaw(cyaw)
    
    di, ai = 0.0, 0.0
    
    while MAX_TIME >= time:
        xref, target_ind, dref = calc_ref_trajectory(
            state, cx, cy, cyaw, ck, sp, dl, target_ind)
    
        x0 = [state.x, state.y, state.v, state.yaw]  # current state

        oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(
            xref, x0, dref, oa, odelta)
        
        # print("ox:         ", ox)
        # print("x reference:", xref[0])
        # print("oy:         ", oy)
        # print("y reference:", xref[1])
        # print("oyaw:       ", oyaw)
        # print("yaw reference:", xref[3])
        # print("ov:         ", ov)
        # print("v reference:", xref[2])
        
        if odelta is not None:
            new_di, ai = odelta[0], oa[0]
            di = smooth_yaw_iter(di, new_di)
            print(f"di:{di}, ai:{ai}")
            # di 大于max，就取max，小于min，就取min
            if di >= MAX_STEER:
                di = MAX_STEER
            elif di <= -MAX_STEER:
                di = -MAX_STEER
            

            state = update_state(state, ai, di)


        time = time + DT

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        d.append(di)
        a.append(ai)

        if check_goal(state, goal, target_ind, len(cx)):
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            if ox is not None:
                plt.plot(ox, oy, "xr", label="MPC")
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plot_car(state.x, state.y, state.yaw, steer=di)
            plt.axis("equal")
            plt.grid(True)
            plt.title("Time[s]:" + str(round(time, 2))
                      + ", speed[km/h]:" + str(round(state.v * 3.6, 2)))
            plt.pause(0.0001)

    return t, x, y, yaw, v, d, a
    x_coords = []
    y_coords = []
    heading_list = []
    speed_list = []
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
            # 将经纬度转换为UTM坐标
            x_coords.append(lon)
            y_coords.append(lat)
            heading_list.append(heading)
            speed_list.append(float(row[3]))
            # 将UTM坐标和航向角存储到traj_data中  
    return x_coords, y_coords, heading_list, speed_list

def main():
    print(__file__ + " start!!")
    start = time.time()

    dl = 1.0  # course tick
    cx, cy, cyaw, ck = read_csv(r'D:\wudi_files\python_files\track_path\PythonRobotics\processed_straight12_17_with_yaw_ck.csv')

    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)  # 主要作用是根据给定的路径规划和控制要求，计算一个适合的速度轨迹
    # 调整初始状态
    # cx[0] = cx[0] + 50
    # cy[0] = cy[0] + 10
    # 初始值
    start_x = cx[0]
    start_y = cy[0]
    print("cyaw 0 :",cyaw[0])
    initial_state = State(x=start_x, y=start_y, yaw=cyaw[0], v=0.0)

    t, x, y, yaw, v, d, a = do_simulation(
        cx, cy, cyaw, ck, sp, dl, initial_state)

    elapsed_time = time.time() - start
    print(f"calc time:{elapsed_time:.6f} [sec]")

    if show_animation:  # pragma: no cover
        plt.close("all")
        plt.subplots()
        plt.plot(cx, cy, "-r", label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        plt.subplots()
        plt.plot(t, v, "-r", label="speed")
        plt.grid(True)
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [kmh]")

        plt.show()


if __name__ == '__main__':
    main()
