"如果转角阈值过大或者过小，做一个滤波"

import os 
from pyproj import Proj, Transformer,transform
import sys
import can
from math import radians, cos, sin, asin, sqrt, degrees, atan2
import matplotlib.pyplot as plt
import time
# from pynput import keyboard  
import csv
import numpy as np
import threading
import math
from read_csv import read_csv
from pynput import keyboard

import math
from read_csv import read_csv
import time

# 车辆参数
VEHICLE_WIDTH = 1.9   # m
VEHICLE_LENGTH = 4.5  # m
WHEEL_FACTOR = 7.2
manual_triggered = False
stop_record = False
mod_666 = 0
mod_AE = 0


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
    def __init__(self, trajectory_csv,target_index=30):
        """
        初始化，读取轨迹点
        :param trajectory_csv: 包含轨迹点的CSV文件路径，轨迹点格式为[经度, 纬度, 航向角度]
        """
        self.trajectory_points = read_csv(trajectory_csv)
        self.closest_index = 0
        self.offset_target_index = target_index
        self.wheelbase = 2.85
        self.target_index = 0
        
        self.previous_turn_angle = 0
        self.max_turn_rate = 6  # 限制每次转向角的最大变化速率（度）

    def init_closest_index(self,current_position, current_heading):
        current_lat, current_lon, _ = current_position
        # 根据后轴的位置和heading调整得到前轴的位置
        front_lat, front_lon = self.adjust_position_to_front_axle(current_lat, current_lon, current_heading)
        
        # 找到距离最近的点的索引
        closest_index = self.find_closest_point_index(front_lat, front_lon)
    
        self.closest_index = closest_index
        print("INIT",self.closest_index)

    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        dLon = math.radians(lon2 - lon1)
        lat1 = math.radians(lat1)
        lat2 = math.radians(lat2)
        x = math.sin(dLon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dLon))
        initial_bearing = math.atan2(x, y)
        initial_bearing = math.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360
        return compass_bearing

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        R = 6371000  # 地球半径，单位为米
        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        a = math.sin(dLat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance

    def find_closest_point_index(self, current_lat, current_lon):
        min_distance = float('inf')
        # closest_index = 0
        for i, (lon, lat, _) in enumerate(self.trajectory_points[self.closest_index:min(self.closest_index+200,len(self.trajectory_points))]):  # 经度在前，纬度在后
            distance = self.calculate_distance(current_lat, current_lon, lat, lon)
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        return closest_index+self.closest_index

    def adjust_position_to_front_axle(self, rear_lat, rear_lon, heading):
        """
        根据后轴中心的经纬度和heading计算前轴的经纬度
        :param rear_lat: 后轴的纬度
        :param rear_lon: 后轴的经度
        :param heading: 车辆的航向角，相对于正北方向
        :return: 前轴的经纬度 (lat, lon)
        """
        # 先将heading转换为弧度
        heading_rad = math.radians(heading)

        # 计算纬度上的变化，假设1度纬度大约为111,320米
        delta_lat = (self.wheelbase / 6371000) * math.cos(heading_rad)

        # 计算经度上的变化，假设经度的变化随着纬度而变化，纬度越高，1度经度的实际距离越小
        delta_lon = (self.wheelbase / 6371000) * math.sin(heading_rad) / math.cos(math.radians(rear_lat))

        # 计算前轴的经纬度
        front_lat = rear_lat + math.degrees(delta_lat)
        front_lon = rear_lon + math.degrees(delta_lon)

        return front_lat, front_lon
    
    def smooth_turn_angle(self, turn_angle):
        # 限制转向角的最大变化速率
        angle_difference = turn_angle - self.previous_turn_angle
        if angle_difference > self.max_turn_rate:
            update_turn_angle = self.previous_turn_angle + 4
        elif angle_difference < -self.max_turn_rate:
            update_turn_angle = self.previous_turn_angle - 4
        else:
            update_turn_angle = turn_angle
        print(f"input:{turn_angle}======>update:{update_turn_angle}")
        # 更新上一次的转向角
        self.previous_turn_angle = update_turn_angle
        return turn_angle

    def calculate_turn_angle(self, current_position, current_heading):

        current_lat, current_lon, _ = current_position
        # 根据后轴的位置和heading调整得到前轴的位置
        front_lat, front_lon = self.adjust_position_to_front_axle(current_lat, current_lon, current_heading)
        
        # 找到距离最近的点的索引
        closest_index = self.find_closest_point_index(front_lat, front_lon)
        self.closest_index = closest_index
        target_index = min(closest_index + self.offset_target_index, len(self.trajectory_points) - 1)  # 防止超出范围
        # print(target_index)
        next_lon, next_lat, _ = self.trajectory_points[target_index]  # 注意经纬度顺序

        # 计算目标点相对当前位置的方位角
        desired_heading = self.calculate_bearing(current_lat, current_lon, next_lat, next_lon)

        # print("==",next_lon, next_lat,current_lon, current_lat,desired_heading)
        # print()
        # 计算转向角
        turn_angle = (desired_heading - current_heading + 360) % 360
        if turn_angle > 180:
            turn_angle -= 360
        
                    
        if turn_angle*WHEEL_FACTOR>380:
            turn_angle = 380
        elif turn_angle*WHEEL_FACTOR<-380:
            turn_angle = -380
        else:
            turn_angle = turn_angle*WHEEL_FACTOR    
        
        turn_angle = self.smooth_turn_angle(turn_angle)
        
        return turn_angle


class Can_use:
    def __init__(self, zone):
        self.bus_ins = can.interface.Bus(channel='can0', bustype='socketcan')
        self.bus_vcu = can.interface.Bus(channel='can1', bustype='socketcan')
        self.ego_lon = 31.8925019
        self.ego_lat = 118.8171577
        self.ego_yaw =  270
        self.ego_v =  3
        self.ego_a = 0
        self.eps_mode = 2
        
    def read_ins_info(self):
        """获取惯导的主车信息"""
        message_ins = self.bus_ins.recv()
        message_vcu = self.bus_vcu.recv()
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

            ego_x = INS_Longitude
            ego_y = INS_Latitude
            self.ego_lon = ego_x
            self.ego_lat = ego_y
            # print(self.ego_lat,self.ego_lon)
             
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

        if message_ins is not None and message_ins.arbitration_id == 0x502:
            # self.ego_yaw = angle
            Angle_data = message_ins.data
            HeadingAngle =  (Angle_data[4] << 8) | Angle_data[5]
            # HeadingAngle =   -(HeadingAngle*0.010986-360-90)
            HeadingAngle =   HeadingAngle*0.010986-360
            self.ego_yaw = HeadingAngle 
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
    can_use.read_ins_info()
    follower.init_closest_index((can_use.ego_lat, can_use.ego_lon, can_use.ego_yaw), can_use.ego_yaw)

    while True:
        can_use.read_ins_info()
        if can_use.eps_mode != 3 and manual_triggered:
            mod_AE = 1
            mod_666 = 1
        if can_use.eps_mode == 3:
            # print("==========================reset============================")
            mod_AE = 3
            mod_666 = 0
            manual_triggered = False
        if mod_AE == 1 and mod_666 == 1:
            if can_use.ego_lon is not None and can_use.ego_lat is not None:

                turn_angle = follower.calculate_turn_angle((can_use.ego_lat, can_use.ego_lon, can_use.ego_yaw), can_use.ego_yaw)
                # print("turn_angle=", turn_angle)
                
                filtered_angle = filter.update_speed(turn_angle)
                new_frame = [5, filtered_angle,0]
                # print(time.time()-t0)
                # if filtered_angle >= 60:
                #     desired_acc = -1
                #     new_frame = [5, filtered_angle,desired_acc]
                # else:
                #     desired_acc = 0
                #     new_frame = [10, filtered_angle,desired_acc]       
            else:
                print("主车定位丢失...")
                new_frame = [-2, 0, 0]
        else:
            print("请按s进入自动驾驶模式...")
            new_frame = [-2, 0, 0]
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
            can_use.publish_planner_ation(action = last_frame, id=0x666, action_type="acc", mod=1, enable=1)

            

def main():
    # 使用示例
    trajectory_csv = '/home/nvidia/vcii/follow_trajectory/collect_trajectory/processed_bigcycle1219.csv'
    # 10
    follower = VehicleTrajectoryFollower(trajectory_csv,target_index=2)
    # 创建过滤器实例
    filter = ISGSpeedFilter()
    # 初始化canbus
    can_use = Can_use(zone=49)
 
    
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

