#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import lib
import numpy as np
import pandas as pd
import time
import math
from planner.plannerBase import PlannerBase
from utils.observation1 import Observation

from utils.opendrive2discretenet import parse_opendrive
from typing import List, Tuple
from pyproj import Proj, Transformer
from shapely.geometry import Point, Polygon
from scipy.optimize import minimize

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

global location  # 给出具体x,y位置
location = [[-1230.349, -383.022],  # 第一个是停车线的位置
            [-1207.684, -443.148],  # 第二个是隧道的位置
            [-514.787, -169.661]]  # 第三个是进环岛的位置

global parking_loc
parking_loc = [[-593.455, -172.784],
               [-593.022, -170.399],
               [-592.802, -167.893]]

class cone_region:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.length = 5
        self.width = 2.5


class IDM(PlannerBase):
    def __init__(self, a_bound=2.0, exv=40, t=1.2, a=2.22, b=2.4, gama=4, s0=1.0, s1=2.0, N=10, dt=0.1, v_max=1,
                 yaw_weight=10.0):
        self.history_line = []
        self.ego = []
        print("start IDM")
        self.a_bound = a_bound
        self.exv = exv
        self.t = t
        self.a = a
        self.b = b
        self.gama = gama
        self.s0 = s0
        self.s1 = s1
        self.s_ = 0
        self.index = 0
        self.all_line = []
        self.ref_line = []
        self.all_line_temp = []
        self.lastWheelAngle = 0
        self.lastAccel = 0
        self.lonlat2xy = Proj('+proj=tmerc +lon_0=108.90575652010739 +lat_0=34.37650478465651 +ellps=WGS84')
        self.N = N  # Prediction horizon
        self.dt = dt  # Time step
        self.v_max = v_max / 3.6  # Maximum velocity in m/s
        self.wait_time = 0
        self.wait_pedestrain = 0
        self.wait_car = 0
        self.pedestrian_stop = False
        self.KP = -2
        self.huandao_time = 0
        self.max_acc = 0.5
        self.end_acc = [0, ]
        self.stopLine = False
        self.half_road = False
        self.jerk_tag = 0
        self.min_hold_on_acc = 0.01
        self.Cross = False
        self.Island = False
        self.Tunnel = False

        self.line = pd.read_csv(
            "/home/pji//onsite-structured-test/all_line.csv").values.tolist()
        self.all_line_temp = self.line

        self.cone_region = cone_region(0, 0)
        self.borrow_obs = []
        self.width = 2.5

        self.p_index = 0

    def init(self, scenario_dict):
        # print("----------------------------IDM* INIT----------------------------")
        print(len(scenario_dict['task_info']['waypoints']))
        self.LineLonLat2XY(scenario_dict)
        # print(len(self.all_line))
        # if (len(self.all_line)!=0): print(self.all_line[0], self.all_line[1])
        # print(self.all_line)
        # print("----------------------------------------------------------------")

    def LineLonLat2XY(self, scenario_dict):
        if len(scenario_dict['task_info']['waypoints']) != 0:
            self.all_line = self.line
            # point = self.generate_point(0)
            # for i in range(len(point)):
            #     self.all_line.append(point[i])
            #     self.all_line_temp.append(point[i])

    def GenerateLaneBorrow(self, obs):
        """lane borrow"""
        # [x, y, width, length]
        if len(obs) == 0:
            return
        self.history_line = []
        index1, index2 = -1, -1
        for i in range(len(self.ref_line)):
            if index1 == -1 and self.ref_line[i][1] >= obs[1] - obs[3] / 2 - 10:
                index1 = i
            if index2 == -1 and self.ref_line[i][1] >= obs[1] + obs[3] / 2 + 10:
                index2 = i
                break
        if index1 == -1 or index2 == -1:
            return
        point0 = [self.ref_line[index1][0], self.ref_line[index1][1]]
        point1 = [obs[0] - self.width, obs[1] - obs[3] / 2]
        point2 = [obs[0] - self.width, obs[1] + obs[3] / 2]
        point3 = [self.ref_line[index2][0], self.ref_line[index2][1]]

        b_line1 = self.generate_bezier(index1, point0, point1)
        b_line2 = self.generate_bezier(index1, point1, point2)
        b_line3 = self.generate_bezier(index1, point2, point3)

        # montage
        b_line_all = []
        b_line_all.extend(b_line1)
        b_line_all.extend(b_line2)
        b_line_all.extend(b_line3)

        for one_s_point in b_line_all:
            new_one_point = self.localxy2utm(one_s_point)
            self.history_line.append([new_one_point[0], new_one_point[1],
                                     one_s_point[2], one_s_point[3]])
        self.all_line_temp = self.all_line_temp[:index1] + self.history_line + self.all_line_temp[index2:]
        # df = pd.DataFrame(self.all_line_temp, columns=['x', 'y', 'yaw', 'v', 'length'])
        # df.to_csv('all_line.csv', index=False)

    def generate_bezier(self, start_index, point0, point1):
        b_line = []
        first_control_point_para_ = 0.3
        second_control_point_para_ = 0.4
        y = point1[1] - point0[1]
        CtrlPointX = [0, 0, 0, 0]
        CtrlPointY = [0, 0, 0, 0]
        CtrlPointX[0] = point0[0]
        CtrlPointY[0] = point0[1]
        CtrlPointX[3] = point1[0]
        CtrlPointY[3] = point1[1]
        CtrlPointX[1] = CtrlPointX[0]
        CtrlPointY[1] = y * first_control_point_para_ + point0[1]
        CtrlPointX[2] = CtrlPointX[3]
        CtrlPointY[2] = y * second_control_point_para_ + point0[1]
        Pos = round(y / 0.5)
        for i in range(Pos, 0, -1):
            tempx = self.bezier3func(i / Pos, CtrlPointX)
            tempy = self.bezier3func(i / Pos, CtrlPointY)
            angle = -1 * (self.cal_angle(i / Pos, CtrlPointX, CtrlPointY) + math.pi / 2) * 180 / math.pi
            angle = self.ref_line[start_index][2] + angle
            if angle > 360:
                angle = angle - 360
            elif angle < 0:
                angle = angle + 360
            b_point = [tempx, tempy, angle, 15, 2]
            b_line.append(b_point)
        return b_line

    def bezier3func(self, _t, controlP):
        part0 = controlP[0] * _t * _t * _t
        part1 = 3 * controlP[1] * _t * _t * (1 - _t)
        part2 = 3 * controlP[2] * _t * (1 - _t) * (1 - _t)
        part3 = controlP[3] * (1 - _t) * (1 - _t) * (1 - _t)
        return part0 + part1 + part2 + part3

    def cal_angle(self, _t, controlP_x, controlP_y):
        _dx_1 = 3 * controlP_x[0] * _t * _t
        _dx_2 = 3 * controlP_x[1] * (_t * 2 - 3 * _t * _t)
        _dx_3 = 3 * controlP_x[2] * (1 - 4 * _t + 3 * _t * _t)
        _dx_4 = -3 * controlP_x[3] * (1 - _t) * (1 - _t)
        _dy_1 = 3 * controlP_y[0] * _t * _t
        _dy_2 = 3 * controlP_y[1] * (_t * 2 - 3 * _t * _t)
        _dy_3 = 3 * controlP_y[2] * (1 - 4 * _t + 3 * _t * _t)
        _dy_4 = -3 * controlP_y[3] * (1 - _t) * (1 - _t)
        return math.atan2(_dy_1 + _dy_2 + _dy_3 + _dy_4, _dx_1 + _dx_2 + _dx_3 + _dx_4)

    def cal_region(self, cone):
        """返回以锥桶坐标为中心的一个长5米宽2.5米的矩形"""
        length = 5
        width = 2.5
        center = [cone[0], cone[1]]
        return length, width, center

    def localxy2utm(self, point):
        x = point[0]
        y = point[1]
        heading_rad = math.radians(90 - self.ego[2])
        dx = x * math.cos(heading_rad) + y * math.sin(heading_rad)
        dy = -x * math.sin(heading_rad) + y * math.cos(heading_rad)
        utm_e = self.ego[0] + dx
        utm_n = self.ego[1] + dy
        return [utm_e, utm_n, point[2], point[3]]

    def parking_planning(self, angle, off_dis):
        self.history_line = []
        self.v_max = 5
        parking_center = parking_loc[self.p_index]
        # 以停车点为中心，绘制一个长为5m, 宽为2.5m的矩形，航向角为angle
        point0 = [parking_center[0] + 2.5 * math.cos(math.radians(180 - angle)),
                  parking_center[1] - 2.5 * math.sin(math.radians(180 - angle))]
        # plt.scatter(point0[0], point0[1], c='g', label='point0')
        point1 = [point0[0] + 1.25 * math.sin(math.radians(180 - angle)),
                  point0[1] + 1.25 * math.cos(math.radians(180 - angle))]
        point2 = [point0[0] - 1.25 * math.sin(math.radians(180 - angle)),
                  point0[1] - 1.25 * math.cos(math.radians(180 - angle))]
        point3 = [point2[0] - 5 * math.cos(math.radians(180 - angle)),
                  point2[1] + 5 * math.sin(math.radians(180 - angle))]
        point4 = [point1[0] - 5 * math.cos(math.radians(180 - angle)),
                  point1[1] + 5 * math.sin(math.radians(180 - angle))]
        plt.plot([point1[0], point2[0], point3[0], point4[0], point1[0]],
                 [point1[1], point2[1], point3[1], point4[1], point1[1]], 'g')
        off_point1 = [parking_center[0] + off_dis * math.cos(math.radians(180 - angle)),
                      parking_center[1] - off_dis * math.sin(math.radians(180 - angle)), angle, 15]
        
        off_point2 = [0, off_dis, self.ego[2], 15]
        ego_point = [0, 0, self.ego[2], 15]
        # plt.scatter(off_point1[0], off_point1[1], c='g', label='off_point1')
        # plt.scatter(off_point2[0], off_point2[1], c='g', label='off_point2')
        plt.plot([x[0] for x in self.all_line_temp], [x[1] for x in self.all_line_temp], 'r')
        parking_center = self.utm2localxy(self.ego[0], self.ego[1], self.ego[2], parking_center[0],
                                          parking_center[1])
        off_point1[0], off_point1[1] = self.utm2localxy(self.ego[0], self.ego[1], self.ego[2], off_point1[0],
                                                        off_point1[1])
        b_line0 = self.interpolate_points([ego_point, off_point2], 0.2)
        for one_s_point in b_line0:
            new_one_point = self.localxy2utm(one_s_point)
            self.history_line.append([new_one_point[0], new_one_point[1]])
        b_line2 = self.interpolate_points([off_point1, parking_center], 0.2)
        b_line1 = self.smooth_connect(b_line0, b_line2, num_points=100)
        for one_s_point in b_line1:
            new_one_point = self.localxy2utm(one_s_point)
            self.history_line.append([new_one_point[0], new_one_point[1]])
        for one_s_point in b_line2:
            new_one_point = self.localxy2utm(one_s_point)
            self.history_line.append([new_one_point[0], new_one_point[1]])
        for i in range(len(self.history_line) - 1):
            dx = self.history_line[i+1][0] - self.history_line[i][0]
            dy = self.history_line[i+1][1] - self.history_line[i][1]
            yaw = self.calculate_angle_between_vectors([1, 0], [dx, dy])
            if dy < 0:
                yaw = 360 - yaw
            self.history_line[i].extend([yaw, 15])
        self.history_line[-1].extend([angle, 15])
        self.all_line_temp = self.history_line
        self.is_parking = True
        # df = pd.DataFrame(self.all_line_temp, columns=['x', 'y', 'yaw', 'v'])
        # df.to_csv('all_line.csv', index=False)

        plt.plot([x[0] for x in self.history_line], [x[1] for x in self.history_line], 'r')
        plt.legend()

    def smooth_connect(self, b_line0, b_line1, num_points=10):
        from scipy.interpolate import splprep, splev
        points = np.array(b_line0 + b_line1)
        x = points[:, 0]
        y = points[:, 1]
        tck, u = splprep([x, y], s=0)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        smoothed_points = [[x_new[i], y_new[i], 0, 0] for i in range(num_points)]
        return smoothed_points

    def interpolate_points(self, points, distance):
        interpolated_points = []
        for i in range(len(points) - 1):
            start_point = np.array([points[i][0], points[i][1]])
            end_point = np.array([points[i + 1][0], points[i + 1][1]])
            diff = end_point - start_point
            num_steps = int(np.linalg.norm(diff) / distance) + 1
            step_size = diff / num_steps
            for j in range(1, num_steps):
                new_point = start_point + j * step_size
                interpolated_points.append([new_point[0], new_point[1], points[i][2], points[i][3], 2.0])
        return interpolated_points

    def act(self, observation: Observation):
        self.UpdateRefLine(observation.ego_info)
        ego_x_y = [observation.ego_info.x, observation.ego_info.y]
        car_lon, car_lat = self.lonlat2xy(observation.ego_info.x, observation.ego_info.y, inverse=True)
        self.ego = [observation.ego_info.x, observation.ego_info.y, observation.ego_info.yaw,
                    observation.ego_info.v, car_lon, car_lat]
        print("ego_x_y: ", [self.ego[0], self.ego[1]])
        if len(self.ref_line) == 0:
            print("no ref line")
            return [0, 0]
        elif len(self.ref_line) > 0:
            self.dev = math.sqrt(self.ref_line[0][0] ** 2 + self.ref_line[0][1] ** 2)
        for i in range(0, len(location)):
            if self.distance(ego_x_y, location[i]) <= 5:
                self.half_road = True
                self.stopLine = True
                if (i == 0):
                    print("Reaching Cross")
                    self.Cross = True
                elif (i == 1):
                    self.Tunnel = True
                    print("Reaching Tunnel")
                elif (i == 2):
                    print("Reaching Island")
                    self.Island = True
                break
            else:
                self.stopLine = False
                self.Cross = False
                self.Island = False
                self.Tunnel = False

        state, surrounding_vehicle = self.get_current_state(observation.ego_info, observation.object_info)
        print(state, surrounding_vehicle)
        # plt.figure()
        # plt.plot([x[0] for x in self.all_line_temp], [x[1] for x in self.all_line_temp], 'b')
        if self.distance(ego_x_y, self.all_line[-1]) <= 10:
            print("Reaching Parking")
            self.parking_planning(173, 10)
        TempLine_v2 = self.GetForwardPoints()
        # 绘制TempLine_v2和self.all_line_temp
        # plt.plot([x[0] for x in TempLine_v2], [x[1] for x in TempLine_v2], 'b')
        # plt.scatter(state[0][0], state[0][1], c='b', label='ego')
        # plt.show()
        # Front_Area_Line = TempLine_v2[0: min(9, len(TempLine_v2))]
        areaDetection = self.generate_band(TempLine_v2, 2.5)
        areaDetection_polygon = self.convert_to_polygon(areaDetection)
        suit_dec = state[0, 2] / self.v_max * self.KP
        control_input = self.mpc_controller(state, TempLine_v2, areaDetection_polygon)

        if len(surrounding_vehicle) != 0:
            data_pedestrian = surrounding_vehicle[surrounding_vehicle[:, 6] == "pedestrian"]
            if len(data_pedestrian) == 0:
                print("no people arrounding")
                data_pedestrian_tag = False
            elif not self.half_road:
                print("people", data_pedestrian)
                data_pedestrian_tag = np.any(surrounding_vehicle[:, 6] == "pedestrian")
            else:
                data_pedestrian_tag = False

            data_car = surrounding_vehicle[surrounding_vehicle[:, 6] == "vehicle"]
            if len(data_car) == 0:
                data_car_tag = False
                print("no car arrounding")
            else:
                print("car", data_car)
                sorted_indices = np.argsort(data_car[:, 7])  # 修改了数据索引
                data_car = data_car[sorted_indices]
                data_car_tag = np.any(surrounding_vehicle[:, 6] == "vehicle")

            data_bycicle = surrounding_vehicle[surrounding_vehicle[:, 6] == "bicycle"]
            if len(data_bycicle) == 0:
                data_bycicle_tag = False
                print("no bycicle arrounding")
            else:
                print("bycicle", data_bycicle)
                sorted_indices = np.argsort(data_bycicle[:, 7])  # 修改了数据索引
                data_bycicle = data_bycicle[sorted_indices]
                data_bycicle_tag = np.any(surrounding_vehicle[:, 6] == "bicycle")

            data_cone = surrounding_vehicle[surrounding_vehicle[:, 6] == "cone"]
            if len(data_cone) == 0:
                print("No cones")
                data_cone_tag = False
            else:
                print("number of cones:", len(data_cone))
                sorted_indices = np.argsort(data_cone[:, 7])
                data_cone = data_cone[sorted_indices]
                data_cone_tag = np.any(surrounding_vehicle[:, 6] == "cone")
                self.cone_region = cone_region(data_cone[0][8], data_cone[0][9])

            if data_cone_tag:  # 恒定优先级
                if abs(self.cone_region.x) < 1 and 15 > self.cone_region.y - self.cone_region.length / 2 > 0:
                    self.borrow_obs = [self.cone_region.x, self.cone_region.y,
                                       self.cone_region.width, self.cone_region.length]
                    # [x, y, width, length]
                    self.GenerateLaneBorrow(self.borrow_obs)
                    # plt.figure()
                    # plt.plot([x[0] for x in self.ref_line], [x[1] for x in self.ref_line], 'b')
                    # plt.plot([x[0] for x in b_line], [x[1] for x in b_line], 'r')
                    # plt.scatter(state[0][0], state[0][1], c='b', label='ego')
                    # plt.scatter(self.cone_region.x, self.cone_region.y, c='g', label='cone')
                    # plt.legend()
                    # plt.show()

            # plt.scatter(data_cone[0][0], data_cone[0][1], c='g', label='cone')
            # plt.plot([x[0] for x in self.all_line_temp], [x[1] for x in self.all_line_temp], 'r')
            # plt.legend()
            # plt.show()

            if data_pedestrian_tag:  # 优先级第一
                if state[0, 2] > 0:
                    if state[0, 2] > 0.1:
                        control_input = [suit_dec, control_input[1]]
                    else:
                        control_input = [-0.1, control_input[1]]
                    self.pedestrian_stop = False
                else:
                    self.pedestrian_stop = True
                    control_input = [0, control_input[1]]
                    self.wait_pedestrain = self.wait_pedestrain + 1

                    # 可以测试时，把这个等待时间取消了，这样就不会有时间限制了
                if self.pedestrian_stop == True:
                    if self.wait_pedestrain < 50:  # stop 7s
                        pass
                    else:
                        self.pedestrian_stop = False
                        control_input = [0.5, control_input[1]]  # start
                ####到这
            elif data_bycicle_tag:  # 优先级第二
                if 0 < (state[0, 2] - data_bycicle[0, 2]) < 0.1:
                    control_input = [0, control_input[1]]
                elif (state[0, 2] - data_bycicle[0, 2]) < 0:
                    pass
                elif (state[0, 2] - data_bycicle[0, 2]) > 1:
                    if state[0, 2] > 0:
                        if state[0, 2] > 0.1:
                            control_input = [suit_dec, control_input[1]]
                        else:
                            control_input = [-0.1, control_input[1]]
                    else:
                        control_input = [0, control_input[1]]
            elif data_car_tag:  # 优先级第三
                if 0 < (state[0, 2] - data_car[0, 2]) < 0.1:
                    control_input = [0, control_input[1]]
                elif 0 > (state[0, 2] - data_car[0, 2]):
                    pass
                elif (state[0, 2] - data_car[0, 2]) > 1:
                    if state[0, 2] > 0:
                        if state[0, 2] > 0.1:
                            control_input = [suit_dec, control_input[1]]
                        else:
                            control_input = [-0.1, control_input[1]]
                    else:
                        control_input = [self.min_hold_on_acc, control_input[1]]  # 这一步很关键
        else:  # 没有任何障碍物
            if state[0, 2] <= 0:
                control_input = [0.5, control_input[1]]
            elif self.v_max >= state[0, 2] >= self.v_max - 0.32:
                control_input = [0, control_input[1]]
        # 自适应设置
        if self.end_acc[-1] * control_input[0] >= 0:
            if control_input[0] - self.end_acc[-1] >= 0.5:
                control_input[0] = self.end_acc[-1] + 0.5
            if control_input[0] >= 0.5:
                control_input[0] = 0.5
            self.jerk_tag = 0
        elif control_input[0] > 0:
            self.jerk_tag = self.jerk_tag + 1
            if self.jerk_tag <= 3:
                if control_input[0] - self.end_acc[-1] >= 0.5:
                    self.end_acc_tmp = self.end_acc[-1] + 0.5
                    control_input[0] = self.end_acc_tmp
                else:
                    pass
            else:
                control_input[0] = 0
        elif control_input[0] < 0:
            self.jerk_tag = self.jerk_tag + 1
            if self.jerk_tag <= 3:
                self.end_acc_tmp = self.end_acc[-1] - 0.5
                control_input[0] = self.end_acc_tmp
            else:
                control_input[0] = 0
            if control_input[0] < -1:
                control_input = -1

        if self.stopLine:
            if self.Island:
                if self.huandao_time < 50:
                    self.huandao_time += 1
                    if state[0, 2] >= 0.1:
                        control_input = [suit_dec, control_input[1]]
                    else:
                        control_input = [-0.1, control_input[1]]
                else:
                    if state[0, 2] < self.v_max:
                        control_input = [0.5, control_input[1]]
                    else:
                        control_input = [0, control_input[1]]
            elif self.Cross:
                if self.huandao_time < 40:
                    self.huandao_time += 1
                    if state[0, 2] >= 0.1:
                        control_input = [suit_dec, control_input[1]]
                    else:
                        control_input = [-0.1, control_input[1]]
                else:
                    if state[0, 2] < self.v_max:
                        control_input = [0.5, control_input[1]]
                    else:
                        control_input = [0, control_input[1]]
            elif self.Tunnel:
                if self.huandao_time < 80:
                    self.huandao_time += 1
                    if state[0, 2] >= 0.1:
                        control_input = [suit_dec, control_input[1]]
                    else:
                        control_input = [-0.1, control_input[1]]
                else:
                    if state[0, 2] < self.v_max:
                        control_input = [0.5, control_input[1]]
                    else:
                        control_input = [0, control_input[1]]
        else:
            self.huandao_time = 0

        control_input = [round(control_input[0], 1), round(control_input[1], 1)]
        self.end_acc.append(control_input[0])
        self.end_acc.pop(0)
        print(control_input)
        return control_input

    def generate_band(self, points, width):
        half_width = width / 2
        left_side = []
        right_side = []

        for i in range(len(points)):
            dx = math.cos(math.radians(points[i][2])) * half_width
            dy = math.sin(math.radians(points[i][2])) * half_width
            left_side.append((points[i][0] - dy, points[i][1] + dx))
            right_side.append((points[i][0] + dy, points[i][1] - dx))

        right_side.reverse()
        band_points = left_side + right_side
        return band_points

    def convert_to_polygon(self, band_points):
        polygon_points = []
        for point in band_points:
            polygon_points.append((point[0], point[1]))
        return Polygon(polygon_points)

    def GetForwardPoints(self):
        ego_x = self.ego[0]
        ego_y = self.ego[1]
        ego_yaw = self.ego[2]
        print("ego_info: ", ego_x, ego_y, ego_yaw)

        def is_forward_point(point):
            dx = point[0] - ego_x
            dy = point[1] - ego_y
            angle = math.atan2(dy, dx) * 180 / math.pi
            angle_diff = (angle - ego_yaw + 360) % 360
            # 将角度差转换到 -180 到 180 范围内
            if angle_diff > 180:
                angle_diff -= 360
            return -90 <= angle_diff <= 90

        # 找到前向点
        forward_points = [point for point in self.all_line_temp if (is_forward_point(point)
                                                                    and self.distance(self.ego, point) < 5)]
        forward_points_index = self.all_line_temp.index(forward_points[0])  # 找到前向的最近点
        self.all_line_temp = self.all_line_temp[forward_points_index:]
        self.ref_line = self.ref_line[forward_points_index:]
        forward_way = self.all_line_temp[0:30]
        # print(len(self.all_line_temp))
        return forward_way

    def UpdateRefLine(self, car):
        self.ref_line = []
        if len(self.all_line) != 0:
            for j in range(len(self.all_line_temp)):
                new_x, new_y = self.utm2localxy(car.x, car.y, car.yaw, self.all_line_temp[j][0],
                                                self.all_line_temp[j][1])
                self.ref_line.append([new_x, new_y, self.all_line_temp[j][2], self.all_line_temp[j][3]])

    def utm2localxy(self, origin_x, origin_y, origin_angle, point_x, point_y):
        det_x = point_x - origin_x
        det_y = point_y - origin_y
        distance = math.sqrt(det_x ** 2 + det_y ** 2)
        angle_line = math.atan2(det_y, det_x) / math.pi * 180
        angle = -1 * (angle_line - origin_angle)
        new_x = distance * math.sin(angle * math.pi / 180)
        new_y = distance * math.cos(angle * math.pi / 180)
        return new_x, new_y

    def calculate_angle_between_vectors(self, v1, v2):
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        # 计算两个向量的模长
        norm_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        norm_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
        # 计算夹角的余弦值
        cos_angle = dot_product / (norm_v1 * norm_v2)
        cos_angle = max(min(cos_angle, 1), -1)
        # 将余弦值转换为角度
        if cos_angle == 1:
            return 0
        elif cos_angle == -1:
            return 180
        else:
            angle = math.acos(cos_angle)
        return math.degrees(angle)

    def get_current_state(self, ego_info, object_info):
        frame = pd.DataFrame(
            vars(ego_info),
            columns=['x', 'y', 'v', 'yaw', 'length', 'width'],
            index=['ego']
        )
        # sub_frame = []
        # ['0x', '1y', '2v', '3yaw', '4length', '5width', '6type', '7distance', '8local_x', '9local_y', '10latt']
        sub_frame = pd.DataFrame()
        for obj_type in object_info:
            for obj_name, obj_info in object_info[obj_type].items():
                obj_x = obj_info.x
                obj_y = obj_info.y
                dx = obj_x - ego_info.x
                dy = obj_y - ego_info.y
                car_angle = [math.cos(math.radians(ego_info.yaw)), math.sin(math.radians(ego_info.yaw))]
                # print(car_angle)
                relative_angle = [dx, dy]
                # tag = car_angle[0]*relative_angle[0] + car_angle[1]*relative_angle[1]
                tag = self.calculate_angle_between_vectors(car_angle, relative_angle)
                if 90 >= tag >= 0:
                    sub_frame_tem = pd.DataFrame(vars(obj_info), columns=['x', 'y', 'v', 'yaw', 'length', 'width'],
                                                 index=[obj_name])
                    sub_frame_tem.insert(loc=sub_frame_tem.shape[1], column="type", value=obj_type)
                    distance = self.distance([frame.x, frame.y], [obj_info.x, obj_info.y])
                    sub_frame_tem.insert(loc=sub_frame_tem.shape[1], column="distance", value=distance)
                    local_x, local_y = self.utm2localxy(ego_info.x, ego_info.y, ego_info.yaw, obj_info.x, obj_info.y)
                    sub_frame_tem.insert(loc=sub_frame_tem.shape[1], column="local_x", value=local_x)
                    sub_frame_tem.insert(loc=sub_frame_tem.shape[1], column="local_y", value=local_y)
                    if obj_info.yaw:
                        angle_diff = abs(obj_info.yaw - self.ego[2])
                    else:
                        angle_diff = 0
                    if_lat = (20 < angle_diff < 160) or (200 < angle_diff < 340)
                    sub_frame_tem.insert(loc=sub_frame_tem.shape[1], column="latt", value=if_lat)
                    if len(sub_frame) == 0:
                        if distance >= 20 or "UNKNOWN" in sub_frame_tem['type']:  # 做了筛选，过滤无法识别的物体
                            pass
                        else:
                            if sub_frame_tem['type'].any() == "pedestrian":
                                sub_frame = sub_frame_tem
                            if if_lat:
                                sub_frame = sub_frame_tem
                            else:
                                if abs(local_x) <= 1:
                                    sub_frame = sub_frame_tem
                    else:
                        if distance >= 20 or "UNKNOWN" in sub_frame_tem['type']:  # 做了筛选，过滤无法识别的物体
                            pass
                        else:
                            if sub_frame_tem['type'].any() == "pedestrian":
                                sub_frame = pd.concat([sub_frame, sub_frame_tem])
                            if if_lat:
                                sub_frame = pd.concat([sub_frame, sub_frame_tem])
                            else:
                                if abs(local_x) <= 1:
                                    sub_frame = pd.concat([sub_frame, sub_frame_tem])
        return frame.to_numpy(), sub_frame.to_numpy()

    def distance(self, ego_info, obj_info):
        det_x = ego_info[0] - obj_info[0]
        det_y = ego_info[1] - obj_info[1]
        dis = math.sqrt(det_x ** 2 + det_y ** 2)
        return dis

    def mpc_controller(self, state, ref_line, areaDetection):
        x0 = np.zeros(2 * self.N)
        # 用车辆当前速度和零转向角作为初始猜测
        # x0[:self.N] = state[0, 2]
        x0[:self.N] = self.lastAccel
        x0[self.N:] = self.lastWheelAngle

        brakeIndicator = False
        for i in range(1, len(state[:, 1])):
            obj_x = state[i][0]
            obj_y = state[i][1]
            point = Point(obj_x, obj_y)
            if areaDetection.contains(point):
                brakeIndicator = True
                break
        # print(123, brakeIndicator)
        if brakeIndicator:
            return [-2, self.lastWheelAngle]

        bounds = [(-self.a_bound, self.a_bound) for _ in range(self.N)] + [(-30, 30) for _ in range(self.N)]
        # 调整优化器的设置，增加迭代次数和容差
        options = {'maxiter': 1000, 'ftol': 1e-4}
        result = minimize(self.mpc_cost_function, x0, args=(state, ref_line), bounds=bounds, method='SLSQP',
                          options=options)

        if result.success:
            a = result.x[0]
            delta = round(result.x[self.N], 1)
            self.lastWheelAngle = delta
            self.lastAccel = a
            # print(1111111111111111111111111111111111111111111111111111111)
            if abs(delta - self.lastWheelAngle) > 0.5:
                if 0 < self.lastWheelAngle < delta:
                    delta = self.lastWheelAngle + 0.5
                elif self.lastWheelAngle > 0 and delta < self.lastWheelAngle:
                    delta = self.lastWheelAngle - 0.5
                elif self.lastWheelAngle < 0 and delta > self.lastWheelAngle:
                    delta = self.lastWheelAngle + 0.5
                elif 0 > self.lastWheelAngle > delta:
                    delta = self.lastWheelAngle - 0.5
            # print(a,delta)
            return [a, delta]

        else:
            print("MPC optimization failed: ", result.message)
            return [self.lastAccel, self.lastWheelAngle]

    def mpc_cost_function(self, control_inputs, state, ref_line):
        cost = 0.0
        x, y, v, yaw = state[0, :4]

        for i in range(self.N):
            a = control_inputs[i]
            delta = control_inputs[self.N + i]
            x, y, v, yaw = self.vehicle_model(x, y, v, yaw, a, delta)
            # print(ref_line)
            ref_x, ref_y, ref_yaw, ref_v = ref_line[min(i, len(ref_line) - 1)]

            yaw_error = abs(yaw - ref_yaw)  # 航向角误差
            yaw_error = min(yaw_error, 10)
            # 代价函数修改系数 -> 看哪个能适应v_max = 25的情况下
            cost += 50 * (x - ref_x) ** 2 + 50 * (
                    y - ref_y) ** 2 + 10 * yaw_error ** 2 + 0.1 * delta ** 2 + 0.1 * a ** 2
            # area4 using this mode  yawweight = 2000/1000 when crossing
            # 增加速度约束的成本
            if v > self.v_max or v < 4:
                cost += 1e2 * (v - self.v_max) ** 2
        return cost

    def vehicle_model(self, x, y, v, yaw, a, delta):
        wheel_base = 2.49
        dt = self.dt
        # 更新速度
        v += a * dt
        v = min(v, self.v_max)  # 速度限制
        # 侧滑角（beta）的计算
        beta_deg = math.degrees(math.atan(0.5 * math.tan(math.radians(delta))))
        # 更新位置
        x += v * math.cos(math.radians(yaw + beta_deg)) * dt
        y += v * math.sin(math.radians(yaw + beta_deg)) * dt
        # 更新航向角
        yaw += (v / wheel_base) * math.sin(math.radians(beta_deg)) * dt
        # 确保航向角在[0, 360]范围内
        yaw = (yaw + 360) % 360
        return x, y, v, yaw

