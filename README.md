# 盒子汽车循迹

## MPC
follow_demo_mpc_bank.py 最好效果版本

## 基于误差算法

目前在学校使用的使用是 follow_demo_2025.py 版本


- follow_demo_v5 和 follow_demo_v6的区别在于：
follow_demo_v5 传进的是障碍物的utm坐标系，然后需要转为经纬度，然后在计算经纬度和轨迹的距离，是否需要调整轨迹
follow_demo_v6 传入的是障碍物的x，y，也就是自车的坐标系下的，直接通过比较y与阈值的大小来判定是否需要改