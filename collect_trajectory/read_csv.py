import csv  
from pyproj import Proj
import matplotlib.pyplot as plt

lonlat2xy_old = Proj('+proj=tmerc +lon_0=118.8170043 +lat_0=31.8926311 +ellps=WGS84')
lonlat2xy_30w = Proj(proj='utm',zone=51,ellps='WGS84', preserve_units='m')
def read_csv(csv_file_path, ego_csv_file):
    traj_data = []
    ego_traj_data = []
    x = []
    y = []
    ego_x = []
    ego_y = []
    # 打开CSV文件并读取内容  
    with open(csv_file_path, mode='r', newline='') as file:  
        csv_reader = csv.reader(file)  
        
        # 跳过标题行（如果有的话）  
        headers = next(csv_reader, None)  # 这行代码会读取第一行，如果第一行是标题则跳过  
        
        # 读取每一行数据并添加到列表中  
        for row in csv_reader:  
            # 将每一行的数据转换为整数或浮点数（根据具体情况选择）  
            # 假设x坐标、y坐标、航向角和速度都是浮点数  
            x_coord = float(row[0])  
            y_coord = float(row[1])
            x_coord_utm, y_coord_utm = lonlat2xy_30w(x_coord, y_coord, inverse=False)  
            heading = float(row[2])  
            speed = float(row[3])  
            x.append(x_coord)
            y.append(y_coord)
            # 将这些信息存储为一个列表，并添加到data_list中  
            data_row = [x_coord, y_coord, heading, speed]  
            traj_data.append(data_row)
    with open(ego_csv_file, mode='r', newline='') as file:  
        csv_reader_ego = csv.reader(file)  
        
        # 跳过标题行（如果有的话）  
        headers = next(csv_reader_ego, None)  # 这行代码会读取第一行，如果第一行是标题则跳过  
        
        # 读取每一行数据并添加到列表中  
        for row in csv_reader_ego:  
            # 将每一行的数据转换为整数或浮点数（根据具体情况选择）  
            # 假设x坐标、y坐标、航向角和速度都是浮点数  
            x_coord = float(row[1]) 
            y_coord = float(row[2])
            x_coord_utm, y_coord_utm = lonlat2xy_30w(x_coord, y_coord, inverse=False)  
            heading = float(row[3])  
            speed = float(row[4])
            if x_coord > 100:  
                ego_x.append(x_coord)
            if y_coord < 100:
                ego_y.append(y_coord)
            # 将这些信息存储为一个列表，并添加到data_list中  
            data_row = [x_coord, y_coord, heading, speed]  
            ego_traj_data.append(data_row)
    # print("ego_x=>", ego_x)
    # print("ego_y=>", ego_y)
    plt.scatter(x , y, color="yellow")
    # plt.scatter([x[-1]], [y[-1]], color="red") # end
    # plt.scatter([x[0]], [y[0]], color="black") # start
    plt.scatter(ego_x, ego_y, color="blue") # ego_traj
    plt.title('reference_trajectory_cycle')  
    plt.xlabel('longitudinal')  
    plt.ylabel('latitudinal')
    plt.savefig('check_reference_2_ego.png')
    return traj_data, ego_traj_data

if __name__ == '__main__':
    read_csv("/home/nvidia/vcii/follow_trajectory/processed_syzx_v2_checked.csv", "/home/nvidia/vcii/follow_trajectory/collect_trajectory/20241009_1302ego.csv")