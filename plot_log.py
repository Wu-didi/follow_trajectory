import re
import matplotlib.pyplot as plt

def extract_angles(log_file_path):
    """
    解析日志文件，提取包含 "trun angle:" 和 "filter angle:" 的行，
    返回 turn_angles 和 filter_angles 两个列表。
    """
    turn_angles = []
    filter_angles = []
    # 匹配类似 "trun angle: 105.75685311262387, filter angle: 88.11959055918209"
    pattern = re.compile(r'trun angle:\s*([\d\.]+),\s*filter angle:\s*([\d\.]+)')
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            match = pattern.search(line)
            if match:
                try:
                    turn_val = float(match.group(1))
                    filter_val = float(match.group(2))
                    turn_angles.append(turn_val)
                    filter_angles.append(filter_val)
                except ValueError:
                    pass
    return turn_angles, filter_angles

def plot_angles(turn_angles, filter_angles, output_file="angle_plot.png"):
    """
    绘制 turn angle 和 filter angle 的折线图，并保存到指定文件
    """
    plt.figure(figsize=(8, 5))
    # 横轴使用数据点的顺序索引
    indices = range(1, len(turn_angles) + 1)
    plt.plot(indices, turn_angles, marker='o', label="turn angle")
    plt.plot(indices, filter_angles, marker='o', label="filter angle")
    plt.xlabel("Sample Index")
    plt.ylabel("Angle (Degrees)")
    plt.title("Turn Angle and Filter Angle Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()
    print(f"图形已保存为 {output_file}")

if __name__ == "__main__":
    log_file = "/home/nvidia/vcii/follow_trajectory/20250327_144525.log"  # 请将此处替换为你的日志文件路径
    turn_angles, filter_angles = extract_angles(log_file)
    print("提取到的 turn angle 数据：", turn_angles)
    print("提取到的 filter angle 数据：", filter_angles)
    
    if turn_angles and filter_angles:
        plot_angles(turn_angles[20000:], filter_angles[20000:])
    else:
        print("没有从日志中找到符合要求的数据。")
