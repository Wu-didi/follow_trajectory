'''生成mpc轨迹跟踪需要的轨迹信息
   输入的是processed_trajectory
   输出处理好的结果
'''

import pandas as pd
import numpy as np
import csv
from pyproj import CRS, Transformer




def latlon_to_utm_zone(latitude, longitude):
    """
    根据经纬度计算所属的UTM投影带号及EPSG代码。
    """
    zone_number = int((longitude + 180) // 6) + 1
    if latitude >= 0:
        epsg_code = f"326{zone_number:02d}"
    else:
        epsg_code = f"327{zone_number:02d}"
    return zone_number, epsg_code

def latlon_to_utm(latitude, longitude):
    """
    将WGS84坐标(经纬度)转换为对应UTM坐标(Easting, Northing)。
    """
    _, utm_epsg = latlon_to_utm_zone(latitude, longitude)
    wgs84 = CRS.from_epsg(4326)
    utm = CRS.from_epsg(int(utm_epsg))
    transformer = Transformer.from_crs(wgs84, utm, always_xy=True)
    easting, northing = transformer.transform(longitude, latitude)
    return easting, northing

def convert_latlon_csv_to_utm_csv(input_csv, output_csv):
    with open(input_csv, mode='r', newline='') as infile, open(output_csv, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ['Easting', 'Northing', 'Heading', 'Speed']  # 输出列字段
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            lon = float(row['Longitude'])
            lat = float(row['Latitude'])
            heading = float(row['Heading'])
            speed = float(row['Speed'])

            # 转换为UTM坐标
            easting, northing = latlon_to_utm(lat, lon)

            out_row = {
                'Easting': easting,
                'Northing': northing,
                'Heading': heading,
                'Speed': speed
            }
            writer.writerow(out_row)





def read_utm_csv(csv_file_path):
    """
    读取包含UTM坐标的CSV文件。

    参数：
    - csv_file_path: CSV文件路径，包含'Easting'和'Northing'列。

    返回：
    - cx: Easting坐标列表
    - cy: Northing坐标列表
    """
    df = pd.read_csv(csv_file_path)
    if not {'Easting', 'Northing'}.issubset(df.columns):
        raise ValueError("CSV文件必须包含'Easting'和'Northing'列。")
    cx = df['Easting'].tolist()
    cy = df['Northing'].tolist()
    return cx, cy

def calculate_yaw_and_curvature(cx, cy):
    """
    根据UTM坐标计算航向角和曲率。

    参数：
    - cx: Easting坐标列表
    - cy: Northing坐标列表

    返回：
    - cyaw: 航向角列表（弧度）
    - ck: 曲率列表
    """
    # 转换为numpy数组
    cx = np.array(cx)
    cy = np.array(cy)
    
    # 计算差分
    dx = np.diff(cx)  # 长度 N-1
    dy = np.diff(cy)  # 长度 N-1
    ds = np.hypot(dx, dy)  # 长度 N-1
    
    # 计算航向角
    cyaw = np.arctan2(dy, dx)  # 长度 N-1
    cyaw = np.append(cyaw, cyaw[-1])  # 长度 N
    
    # 计算一阶导数
    dx_ds = dx / ds  # 长度 N-1
    dy_ds = dy / ds  # 长度 N-1
    
    # 计算二阶导数
    d2x_ds2 = np.diff(dx_ds) / ds[:-1]  # 长度 N-2
    d2y_ds2 = np.diff(dy_ds) / ds[:-1]  # 长度 N-2
    
    # 计算曲率
    numerator = dx_ds[:-1] * d2y_ds2 - dy_ds[:-1] * d2x_ds2  # 长度 N-2
    denominator = (dx_ds[:-1]**2 + dy_ds[:-1]**2)**1.5  # 长度 N-2
    ck_inner = numerator / denominator  # 长度 N-2
    
    # 初始化曲率数组
    ck = np.zeros_like(cx)  # 长度 N
    
    # 填充中间的曲率值
    ck[1:-1] = ck_inner
    
    # 填充首尾的曲率值为相邻的曲率
    ck[0] = ck[1]
    ck[-1] = ck[-2]
    
    # 处理可能的NaN或无穷大值
    ck = np.where(np.isfinite(ck), ck, 0.0)
    
    # 转换为列表
    cyaw = cyaw.tolist()
    ck = ck.tolist()
    
    return cyaw, ck

def get_course(csv_file_path):
    """
    读取UTM坐标CSV文件并计算航向角和曲率。

    参数：
    - csv_file_path: 输入的UTM坐标CSV文件路径。

    返回：
    - cx: Easting坐标列表
    - cy: Northing坐标列表
    - cyaw: 航向角列表（弧度）
    - ck: 曲率列表
    """
    # 读取UTM坐标
    cx, cy = read_utm_csv(csv_file_path)
    
    # 计算航向角和曲率
    cyaw, ck = calculate_yaw_and_curvature(cx, cy)
    
    return cx, cy, cyaw, ck

def main():
    
    input_file        = "/home/nvidia/vcii/follow_trajectory/collect_trajectory/processed_shiyanzhongxin_0327.csv"
    temp_output_file  = "/home/nvidia/vcii/follow_trajectory/collect_trajectory/processed_shiyanzhongxin_0327_utm.csv"
    final_output_file = "/home/nvidia/vcii/follow_trajectory/collect_trajectory/processed_shiyanzhongxin_0327_with_yaw_ck.csv"
    convert_latlon_csv_to_utm_csv(input_file, temp_output_file)
    print("转换完成！")
    # 替换为您的UTM坐标CSV文件路径
    input_csv = temp_output_file
    
    # 获取cx, cy, cyaw, ck
    cx, cy, cyaw, ck = get_course(input_csv)
    
    # 打印部分结果
    for i in range(min(10, len(cx))):
        print(f"Point {i}: cx={cx[i]}, cy={cy[i]}, cyaw={cyaw[i]:.4f} rad, ck={ck[i]:.6f} 1/m")
    
    # 如果需要，可以将结果保存为新的CSV文件
    output_df = pd.DataFrame({
        'cx': cx,
        'cy': cy,
        'cyaw': cyaw,
        'ck': ck
    })
    output_df.to_csv("", index=False)
    print("计算完成，结果已保存至 'trajectory_with_yaw_ck.csv'")

if __name__ == "__main__":
    main()
