import os 
from pyproj import Proj, Transformer,transform
from math import radians, cos, sin, asin, sqrt, degrees, atan2
import matplotlib.pyplot as plt
import time
from pynput import keyboard  
import csv
import numpy as np
import threading
import math
import time  
import can

class Can_use:
    def __init__(self, zone):
        self.bus_ins = can.interface.Bus(channel='can0', bustype='socketcan')
        self.bus_vcu = can.interface.Bus(channel='can1', bustype='socketcan')
        self.zone = zone
        # 将经纬度转化为utm坐标
        self.lonlat2xy = Proj(proj='utm', zone=49, ellps='WGS84', preserve_units='m')
        self.lonlat2xy_old = Proj('+proj=tmerc +lon_0=118.8170043 +lat_0=31.8926311 +ellps=WGS84')
        # self.lonlat2xy_old = Proj('+proj=tmerc +lon_0=108.90575652010739 +lat_0=34.37650478465651 +ellps=WGS84')
        # 主车lon
        self.ego_lon = None
        # 主车lat
        self.ego_lat = None
        # 主车velocity
        self.ego_v = 0
        # 主车acc
        self.ego_a = 0
        # 主车yaw
        self.ego_yaw = 0
        # 是否进入自动驾驶模式
        self.auto_driver_allowed = False
        # 键盘哨兵
        self.key_guard = False

    def read_ins_info(self, mode):
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
            # 打印经纬度信息
            # print(f"接收到消息 ID=0x{message.arbitration_id:X}, 纬度: {INS_Latitude}, 经度: {INS_Longitude}")
            # print("lon=", INS_Longitude)
            ego_x = INS_Longitude
            ego_y = INS_Latitude
            self.ego_lon = ego_x
            self.ego_lat = ego_y
             
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
            # if HeadingAngle < 0:
            #     HeadingAngle += 360
            HeadingAngle =   HeadingAngle*0.010986-360

            self.ego_yaw = HeadingAngle 
            # print("====================================",angle)
            # print("===============speed=============:",speed)
            # print(f"INS_NorthSpd: {INS_NorthSpd*3.6}, INS_EastSpd: {INS_EastSpd*3.6}, INS_ToGroundSpd: {INS_ToGroundSpd*3.6}")
        if message_ins is not None and message_ins.arbitration_id == 0x500:
            acc_data = message_ins.data
            # 北向速度
            ACC_X =  (acc_data[0] << 8) | acc_data[1]
            ACC_X =   (ACC_X*0.0001220703125-4)*9.8   # g
            self.ego_a = ACC_X
            # print("=============================ACC_X:", ACC_X)
        
        if message_vcu is not None and message_vcu.arbitration_id == 0x15C:
            allow_value = message_vcu.data[2] & 0x01
            self.auto_driver_allowed = (allow_value == 1)

        # if keyboard.is_pressed('s'):
        #     # 若键盘键入s，则启动自动驾驶模式
        #     self.auto_driver_allowed = True
        # if keyboard.is_pressed('esc'):
        #     # 若键盘键入esc，则退出自动驾驶模式
        #     self.auto_driver_allowed = False

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
            data3 = int(100 / 10) & 0xFF     # data3缩放到8位范围，0-255, angle_spd=100
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
            speed_scaled = int(1) & 0x1F  # 取5位（位3-7）
            # 组合BYTE0
            byte0 = (speed_scaled << 3) | auto_drive_cmd_bits

            # BYTE1-BYTE2（需求方向盘转角）
            # 需要根据具体缩放因子和偏移量进行计算，假设缩放因子为0.1，偏移量为0
            angle_scaled = int((180 - (-180)) / 0.01) & 0xFFFF  # 16位
            byte1 = (angle_scaled >> 8) & 0xFF  # 高8位
            byte2 = angle_scaled & 0xFF         # 低8位

            # BYTE3（需求制动减速度）
            # 进行缩放和偏移
            acc_scaled = int((3 - (-4)) / 1) & 0xFF  # 假设缩放因子1，偏移量-4

            # 构建发送数据，剩余字节填充0
            data_666 = [byte0, byte1, byte2, acc_scaled, 0, 0, 0, 0]
            # print("acc id", id)
            
            msg = can.Message(arbitration_id=id, data=data_666, is_extended_id=False)
            # 发送CAN消息
            self.bus_vcu.send(msg)
            # time.sleep(0.01)

        # 限制发送频率
        # time.sleep(0.01)
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """计算两个经纬度点之间的直线距离"""
        # 将十进制度数转化为弧度  
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])  
    
        # haversine公式  
        dlat = lat2 - lat1  
        dlon = lon2 - lon1  
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2  
        c = 2 * asin(sqrt(a))  
        r = 6371000  # 地球平均半径，单位为公里  
        return c * r 

stop_record = False

def on_press(key):
    global manual_triggered
    global stop_record
    try:
        if key.char == 's':
            manual_triggered = True
            # print("收到键盘输入's'，手动请求自动驾驶模式")
        if key.char == 'q':
            manual_triggered = False
        if key.char == "x":
            stop_record = True
    except AttributeError:
        if key == keyboard.Key.esc:
            print("收到Esc键，退出程序")
            return False  # 停止监听


def start_keyboard_listener():  
    # 创建并启动键盘监听器线程  
    with keyboard.Listener(on_press=on_press) as listener:  
        listener.join() 
    
def adjust_position_to_front_axle(rear_lat, rear_lon, heading):
    """
    根据后轴中心的经纬度和heading计算前轴的经纬度
    :param rear_lat: 后轴的纬度
    :param rear_lon: 后轴的经度
    :param heading: 车辆的航向角，相对于正北方向
    :return: 前轴的经纬度 (lat, lon)
    """
    wheelbase = 2.85
    # 先将heading转换为弧度
    heading_rad = math.radians(heading)

    # 计算纬度上的变化，假设1度纬度大约为111,320米
    delta_lat = (wheelbase / 6371000) * math.cos(heading_rad)

    # 计算经度上的变化，假设经度的变化随着纬度而变化，纬度越高，1度经度的实际距离越小
    delta_lon = (wheelbase / 6371000) * math.sin(heading_rad) / math.cos(math.radians(rear_lat))

    # 计算前轴的经纬度
    front_lat = rear_lat + math.degrees(delta_lat)
    front_lon = rear_lon + math.degrees(delta_lon)

    return front_lat, front_lon

def start_record(interval, name):
    global stop_record
    print("start record ref_traj!")
    mode = "recorder"
    frames = 0
    # 初始化canbus
    can_use = Can_use(zone=50)
    ref_traj = []
    start_point = []
    ego_info_last = np.zeros(6)
    ego_info_current =np.zeros(6)

    traj_path = name + ".csv"
    traj_figure = name + ".png"
    while True:
        # lon, lat, yaw, velocity
        traj_point = [] 
        frames += 1
        can_use.read_ins_info(mode)
        if can_use.ego_lon is not None and can_use.ego_lat is not None:
            ego_info_current[0] = frames
            ego_info_current[1] = round(can_use.ego_lon, 10)
            ego_info_current[2] = round(can_use.ego_lat, 10)
            ego_info_current[3] = can_use.ego_v
            ego_info_current[4] = can_use.ego_a
            ego_info_current[5] = can_use.ego_yaw
            # ego_info_current[2],ego_info_current[1] = adjust_position_to_front_axle(ego_info_current[2],ego_info_current[1],can_use.ego_yaw)
            traj_point.append(ego_info_current[1]) 
            traj_point.append(ego_info_current[2])
            traj_point.append(ego_info_current[5])
            traj_point.append(ego_info_current[3])
            ref_traj.append(traj_point)
            start_point = ref_traj[0]
            ego_info_last = ego_info_current
            print("stop_record=", stop_record)
            if stop_record:
                # 绘制轨迹
                x = []
                y = []
                for point in ref_traj:
                    x.append(point[0])
                    y.append(point[1])
                x = np.array(x)
                y = np.array(y)
                plt.scatter(x, y, color="blue")
                plt.scatter([x[-1]], [y[-1]], color="red") # end
                plt.scatter([x[0]], [y[0]], color="black") # start
                plt.title('reference_trajectory')  
                plt.xlabel('longitudinal')  
                plt.ylabel('latitudinal')
                plt.savefig(traj_figure)
                with open(traj_path, mode='w', newline='') as file:  
                    writer = csv.writer(file)  
                    writer.writerows(ref_traj)
            print("ego_info=>", ego_info_last, len(ego_info_last))

if __name__ == '__main__':
    mode = "recorder"
    interval = 0.2
    name = "shiyanzhongxin_0327"
    # 监听是否停止采集
    keyboard_thread = threading.Thread(target=start_keyboard_listener, daemon=True)
    keyboard_thread.start()
    start_record(interval, name)