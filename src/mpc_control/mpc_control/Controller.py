import numpy as np

def controller(agent, u, n):
    """
    根据当前智能体状态和 jerk 输入 u=[jx,jy,jz]，用离散积分更新到下一拍。
    返回:
      x : 长度 12 的状态 [x,vx,ax, y,vy,ay, z,vz,az, yaw, roll, pitch]
      v : 标量线速度 ||v||
      w : 标量角速度幅值 ||omega||（标量）
    说明:
      - 所有参与运算的量都转为 numpy.float，以避免 list 与 float 运算报错
      - 只使用 agent.theta 的最后 3 个分量（yaw, roll, pitch）
    """
    # ===== 基本参数 =====
    dt = 1.0 / float(agent.controlFrequency)

    # 统一规范输入为 numpy 浮点向量
    u = np.asarray(u, dtype=float).reshape(3,)                       # jerk [jx, jy, jz]
    lastAcce = np.asarray(agent.acceleration, dtype=float).reshape(3,)
    lastVelo = np.asarray(agent.velocity,     dtype=float).reshape(3,)
    currentPosition = np.asarray(agent.position, dtype=float).reshape(3,)

    # theta 只用最后 3 个分量，不足则补 0
    theta_raw = np.asarray(agent.theta, dtype=float).reshape(-1)
    if theta_raw.size >= 3:
        theta = theta_raw[-3:]
    else:
        theta = np.pad(theta_raw, (0, 3 - theta_raw.size)).astype(float)

    # 速度/角速度限幅
    bound  = 3.0                   # 角速度上限（幅值）
    vbound = 0.02                  # 最小线速度
    vmax   = float(agent.vmax) * 0.99

    # ===== jerk 模型离散积分 =====
    # v_{k+1} = v_k + a_k dt + 0.5 * j dt^2
    v_vec_next = lastVelo + lastAcce * dt + 0.5 * u * dt**2
    # a_{k+1} = a_k + j dt
    a_vec_next = lastAcce + u * dt
    # x_{k+1} = x_k + v_k dt + 0.5 a_k dt^2 + (1/6) j dt^3
    pos_next   = currentPosition + lastVelo * dt + 0.5 * lastAcce * dt**2 + (1.0/6.0) * u * dt**3

    # ===== 角速度 =====
    # 曲率公式 omega=(v×a)/||v||^2 针对轮式底盘，四旋翼不适用；MulticopterVelocityControl 下直接置零
    # 可选：若需机头朝向飞行方向，可启用下方 Yaw P 控制
    omega_vec = np.array([0.0, 0.0, 0.0], dtype=float)
    # if float(np.linalg.norm(v_vec_next[:2])) > 0.1:
    #     target_yaw = np.arctan2(v_vec_next[1], v_vec_next[0])
    #     yaw_error = target_yaw - theta[0]
    #     yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi
    #     omega_vec[2] = 1.0 * yaw_error
    omega_mag = float(np.linalg.norm(omega_vec))

    # ===== 速度限幅（修复：真正修改速度向量，而非仅改标量 v_mag）=====
    v_mag = float(np.linalg.norm(v_vec_next))
    if v_mag > 1e-9 and v_mag < vbound:
        v_vec_next = v_vec_next / v_mag * vbound
    elif v_mag > vmax:
        v_vec_next = v_vec_next / v_mag * vmax
    v_mag = float(np.linalg.norm(v_vec_next))  # 限幅后重新计算，供返回值使用

    # ===== 姿态更新（逐轴积分）=====
    theta_next = theta + omega_vec * dt

    # ===== 加速度限幅（逐分量）=====
    a_vec_next = np.clip(a_vec_next, -0.99, 0.99)

    # ===== 组装 12 维状态向量 =====
    x = np.array([
        pos_next[0], v_vec_next[0], a_vec_next[0],
        pos_next[1], v_vec_next[1], a_vec_next[1],
        pos_next[2], v_vec_next[2], a_vec_next[2],
        theta_next[0], theta_next[1], theta_next[2]
    ], dtype=float)

    # （可选）同步回写 agent，便于外部直接读取
    agent.position     = pos_next
    agent.velocity     = v_vec_next
    agent.acceleration = a_vec_next
    agent.theta        = theta_next
    agent.angular_velocity = omega_vec  # 供 ROS2 Twist.angular 使用

    # 返回：状态、标量线速度、标量角速度幅值
    return x, v_mag, omega_mag