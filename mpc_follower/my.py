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


def angle_mod(x, zero_2_2pi=False, degree=False):
    """
    Angle modulo operation
    Default angle modulo range is [-pi, pi)

    Parameters
    ----------
    x : float or array_like
        A angle or an array of angles. This array is flattened for
        the calculation. When an angle is provided, a float angle is returned.
    zero_2_2pi : bool, optional
        Change angle modulo range to [0, 2pi)
        Default is False.
    degree : bool, optional
        If True, then the given angles are assumed to be in degrees.
        Default is False.

    Returns
    -------
    ret : float or ndarray
        an angle or an array of modulated angle.

    """
    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle


def smooth_yaw_iter(previous_yaw, new_yaw):
    """
    Smooth the yaw angle based on the previous yaw to ensure continuity.

    :param previous_yaw: (float) Previous yaw angle in radians
    :param new_yaw: (float) New yaw angle in radians (not yet normalized)
    :return: (float) Smoothed and normalized yaw angle in radians within (-pi, pi]
    """
    dyaw = new_yaw - previous_yaw

    # 调整dyaw，使其在[-pi, pi]范围内
    dyaw = (dyaw + np.pi) % (2.0 * np.pi) - np.pi

    # 平滑后的yaw
    smoothed_yaw = previous_yaw + dyaw
    # print(f"previous_yaw:{previous_yaw}, new_yaw:{new_yaw}, smoothed_yaw:{smoothed_yaw}")
    return smoothed_yaw


class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None


def pi_2_pi(angle):
    return angle_mod(angle)


def get_linear_model_matrix(v, phi, delta):
    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = - DT * v * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB

    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(phi) * phi
    C[1] = - DT * v * math.cos(phi) * phi
    C[3] = - DT * v * delta / (WB * math.cos(delta) ** 2)

    return A, B, C


def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD,
                          -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")


def normalize_angle(angle):
    """
    Normalize an angle to (-pi, pi].

    :param angle: (float) Angle in radians
    :return: (float) Angle in radians within (-pi, pi]
    """
    normalized_angle = (angle + np.pi) % (2.0 * np.pi) - np.pi
    if normalized_angle <= -np.pi:
        normalized_angle += 2.0 * np.pi
    return normalized_angle


def update_state(state, a, delta):
    # input check
    if delta >= MAX_STEER:
        delta = MAX_STEER
    elif delta <= -MAX_STEER:
        delta = -MAX_STEER

    state.x = state.x + state.v * math.cos(state.yaw) * DT
    state.y = state.y + state.v * math.sin(state.yaw) * DT
    state.yaw = state.yaw + state.v / WB * math.tan(delta) * DT
    # state.yaw = normalize_angle(state.yaw)
    state.v = state.v + a * DT

    if state.v > MAX_SPEED:
        state.v = MAX_SPEED
    elif state.v < MIN_SPEED:
        state.v = MIN_SPEED

    return state


def get_nparray_from_matrix(x):
    return np.array(x).flatten()


def calc_nearest_index(state, cx, cy, cyaw, pind):
    # print(len(cx[pind:(pind + N_IND_SEARCH)]))
    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind) + pind

    mind = math.sqrt(mind)  # 最小距离

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind


def predict_motion(x0, oa, od, xref):
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
    for (ai, di, i) in zip(oa, od, range(1, T + 1)):
        state = update_state(state, ai, di)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.v
        xbar[3, i] = state.yaw

    return xbar


def iterative_linear_mpc_control(xref, x0, dref, oa, od):
    """
    MPC control with updating operational point iteratively
    """
    ox, oy, oyaw, ov = None, None, None, None

    if oa is None or od is None:
        oa = [0.0] * T  # 上一轮优化得到的加速度序列，如果是第一次迭代，就初始化为0
        od = [0.0] * T  # 上一轮优化得到的转角序列，如果是第一次迭代，就初始化为0

    for i in range(MAX_ITER):
        xbar = predict_motion(x0, oa, od, xref)
        poa, pod = oa[:], od[:]
        oa, od, ox, oy, oyaw, ov = linear_mpc_control(xref, xbar, x0, dref)
        du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value 用于判断是否收敛
        if du <= DU_TH:  # 如果u的变化小于阈值，说明收敛了, 就退出迭代
            break
    else:
        print("Iterative is max iter")

    return oa, od, ox, oy, oyaw, ov


def linear_mpc_control(xref, xbar, x0, dref):
    """
    linear mpc control

    xref: reference point
    xbar: operational point
    x0: initial state
    dref: reference steer angle
    """

    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t] - dref[:, t], R)  # cost function 控制输入的cost

        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)  # state cost function

        A, B, C = get_linear_model_matrix(
            xbar[2, t], xbar[3, t], dref[0, t])
        constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]  # 动力学约束 保证车辆的运动符合动力学模型

        if t < (T - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t],
                                    Rd)  # input difference cost function 控制输入的变化cost，目的是让控制输入尽量平滑，减小震荡
            constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <=  # 控制输入的变化率约束，限制转角的变化率，防止转角变化过快
                            MAX_DSTEER * DT]

    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)  # final state cost function 终端状态的cost

    constraints += [x[:, 0] == x0]
    constraints += [x[2, :] <= MAX_SPEED]  # 速度约束
    constraints += [x[2, :] >= MIN_SPEED]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]  # 加速度约束
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.CLARABEL, verbose=False)
    # prob.solve(solver=cvxpy.ECOS, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox = get_nparray_from_matrix(x.value[0, :])
        oy = get_nparray_from_matrix(x.value[1, :])
        ov = get_nparray_from_matrix(x.value[2, :])
        oyaw = get_nparray_from_matrix(x.value[3, :])
        oa = get_nparray_from_matrix(u.value[0, :])
        odelta = get_nparray_from_matrix(u.value[1, :])

    else:
        print("Error: Cannot solve mpc..")
        oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

    return oa, odelta, ox, oy, oyaw, ov


def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, pind):
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((NU, T + 1))
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)
    # print("最近点的索引：", ind, pind)
    if pind >= ind:
        ind = pind

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = sp[ind]
    xref[3, 0] = cyaw[ind]
    dref[0, 0] = math.atan2(WB * ck[ind], 1)  # steer operational point should be 0

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
            dref[0, i] = math.atan2(WB * ck[ind + i], 1)
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = sp[ncourse - 1]
            xref[3, i] = cyaw[ncourse - 1]
            dref[0, i] = math.atan2(WB * ck[ncourse - 1], 1)

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


def calc_speed_profile(cx, cy, cyaw, target_speed):
    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile


def smooth_yaw(yaw):
    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw


def read_csv(csv_file_path):
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
    cx, cy, cyaw, ck = read_csv(
        r'/home/nvidia/vcii/follow_trajectory/collect_trajectory/processed_wandao1219_with_yaw_ck.csv')

    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)  # 主要作用是根据给定的路径规划和控制要求，计算一个适合的速度轨迹
    # 调整初始状态
    # cx[0] = cx[0] + 50
    # cy[0] = cy[0] + 10
    # 初始值
    start_x = cx[0]
    start_y = cy[0]
    print("cyaw 0 :", cyaw[0])
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
