import csv  
import math  
from pyproj import Proj, Transformer,transform
# # 常量定义  
# EARTH_RADIUS = 6371000  # 地球半径，单位：米  
# METERS_TO_DEGREES_LAT = 180 / (math.pi * EARTH_RADIUS)  # 纬度上1米对应的度数  
# METERS_TO_DEGREES_LON = METERS_TO_DEGREES_LAT * math.cos(math.radians(45))  # 假设中等纬度，经度上1米对应的度数（近似）  
# # 注意：对于更精确的计算，应该根据每个点的纬度来计算METERS_TO_DEGREES_LON  

lonlat2xy = Proj(proj='utm', zone=50, ellps='WGS84', preserve_units='m')
def utm_dis(lon1, lat1, lon2, lat2):  
   x1, y1 = lonlat2xy(lon1, lat1, inverse=False)
   x2, y2 = lonlat2xy(lon2, lat2, inverse=False)
   dis = math.sqrt((x1 - x2)**2 + (y1-y2)**2)
   return dis
  
def process_vehicle_trajectory(input_file, output_file, distance_threshold=1):  
    with open(input_file, mode='r', newline='') as infile, open(output_file, mode='w', newline='') as outfile:  
          
        reader = csv.reader(infile)  
        writer = csv.writer(outfile)  
          
        # 写入表头  
        writer.writerow(['Longitude', 'Latitude', 'Heading', 'Speed'])  
          
        previous_point = None  
        for row in reader:  
            lon, lat, heading, speed = map(float, row)  
              
            if previous_point is not None:  
                prev_lon, prev_lat = previous_point  
                distance = utm_dis(prev_lon, prev_lat, lon, lat) 
                  
                # 如果距离小于阈值，则跳过当前点  
                if distance < distance_threshold:
                    continue 
            # 写入当前点  
            writer.writerow([lon, lat, heading, speed])  
              
            # 更新previous_point  
            previous_point = (lon, lat)  
      
    print(f"Processing complete. Points with distance less than {distance_threshold}m have been skipped.")  

input_csv = '/home/nvidia/vcii/follow_trajectory/collect_trajectory/shiyanzhongxin_0327.csv'  
output_csv = '/home/nvidia/vcii/follow_trajectory/collect_trajectory/processed_shiyanzhongxin_0327.csv'  
process_vehicle_trajectory(input_csv, output_csv)

