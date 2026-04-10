import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import dblquad
from scipy import optimize
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from scipy.optimize import linprog
from sklearn.mixture import GaussianMixture
import time

from shapely.geometry import LineString, Point
from qpsolvers import solve_qp
from shapely.geometry import Polygon as ShapelyPolygon

# ---------------------------------------------------------------------------
# APF 宏观调参：优先平滑飞向 GMM 目标，减弱机间/边界与吸引竞争导致的振荡、乱飞
# ---------------------------------------------------------------------------
# 吸引：协方差平滑越大，远处梯度越温和、易沿主方向飞向目标
APF_SIGMA_K_DIAG = 0.45
# 吸引梯度增益（相对原实现放大，避免被斥力“压过”）
APF_ATTRACT_GAIN = 2.2
# 机间斥力（高斯 + 反比距离）整体缩放；原 r_repulsion=0.16m 时多机极易互相弹开
APF_AGENT_REPULSE_GAIN = 0.28
# 障碍物 ESDF 斥力缩放（略低更易过窄缝；与下方「单位向量混合」一起用）
APF_ESDF_REPULSE_GAIN = 0.32
# FIESTA 的 grad 为 ∇d（距离增大方向、背离障碍）。步长 pos -= dU_unit*v 为下坡，
# 障碍势能近障更高，∇U_obstacle ∝ −∇d，故 dU 中应加「−∇d」而非「+∇d」。
# 斥力公式里 d 过小会数值爆炸，做下限；混合权重避免「全向量相加再归一」时斥力完全盖住目标方向。
APF_ESDF_SAFE_DIST_MIN = 0.08
APF_ATT_REP_UNIT_BLEND = 0.68
# 地图边界斥力缩放（靠边时原力极易饱和引发抖动）
APF_BOUNDARY_REPULSE_GAIN = 0.35
# 机间势能在标量 U / U_next 中的权重；须与 U_next 一致，否则外层 APF 迭代不收敛、轨迹抖
APF_GAMMA_INTER_AGENT = 0.22
# 归一化方向后每子步最大位移（米）；外层 APF 会多次调用，略降基底可减过冲
APF_MAX_VELOCITY_BASE = 0.78  # APF 子步位移上限 [m]，与 MPC v_max 提高配套


def find_nonzero_elements(arr):
    nonzero_indices = []
    nonzero = []
    for i, element in enumerate(arr):
        if element != 0:
            nonzero_indices.append(i)
            nonzero.append(element)
    return len(nonzero_indices), nonzero_indices, nonzero


def Wasserstein_distance(mean1, cov1, mean2, cov2):
    # Calculate the Wasserstein distance between two multivariate normal distributions

    mean1 = np.array(mean1)
    mean2 = np.array(mean2)
    cov1 = np.array(cov1)
    cov2 = np.array(cov2)
    add1 = np.linalg.norm(mean1 - mean2)
    if np.array_equal(cov1, cov2):
        W = add1
    else:
        add2 = np.trace(cov1 + cov2 - 2 * sqrtm(sqrtm(cov1) @ cov2 @ sqrtm(cov1)))
        W = (add1 ** 2 + add2) ** 0.5
    return W


def calWGMetric_speedUp(means1, covs1, weights1, means2, covs2, weights2):
    NumComp_p, NumComp_q = len(means1), len(means2)

    # 自动归一化权重以确保传输质量匹配
    total_mass = min(sum(weights1), sum(weights2))
    weights1 = [w / sum(weights1) * total_mass for w in weights1]
    weights2 = [w / sum(weights2) * total_mass for w in weights2]

    C = np.zeros((NumComp_p, NumComp_q), dtype=float)
    for i in range(NumComp_p):
        for j in range(NumComp_q):
            C[i, j] = Wasserstein_distance(means1[i], covs1[i], means2[j], covs2[j]) ** 2
    f = C.T.flatten()

    Aeq = np.zeros((NumComp_p + NumComp_q, NumComp_p * NumComp_q), dtype=float )
    for i in range(NumComp_q):
        Aeq[i, i * NumComp_p: (i + 1) * NumComp_p] = 1
    for i in range(NumComp_q, NumComp_p + NumComp_q):
        Aeq[i, i - NumComp_q::NumComp_p] = 1
    beq = np.array(weights2 + weights1)

    result = linprog(f, A_eq=Aeq, b_eq=beq, bounds=[(0, 1)] * (NumComp_p * NumComp_q), method='highs-ds')
    
    if not result.success:
        print("❌ LP failed:", result.message)
        raise RuntimeError("Linear programming did not succeed.")

    W, fval = result.x, result.fun
    WG_sq = fval
    WG = np.sqrt(fval)

    return WG_sq, W, WG


def risk_term_F(gimean, gjmean, gicov, gjcov, obstacle_vertices):
    def R(gmean, gcov, obstacle_vertices):
        value = ([gmean], gcov, obstacle_vertices, 0.1)
        return value

    result = (R(gimean, gicov, obstacle_vertices) + R(gjmean, gjcov, obstacle_vertices)) / 2
    return result


def APF(next_means, next_covs, next_weights, robots_positions, esdf_map, MaxNumTry=10):  # 0.04s
    n = 1
    J_rate = float('inf')           #when J_rate is little enough,dif_J_rate is close to 0, the algorithm converge
    J_rate_pre = float('inf')
    dif_J_rate = float('inf')
    J_next = float('inf')          
    diff_gmm_est_targ = []
    robots_positions_list = []
    while n <= MaxNumTry and J_rate > 1e-6:
        start = time.time()
        robots_positions, J_rate, J, J_next = agentControl_APF(next_means, next_covs, next_weights, robots_positions,
                                                               esdf_map, MaxNumTry)
        #print(f'agentControl_APF{time.time() - start}')
        robots_positions_list.append(robots_positions)
        '''
        current_means, current_covs, current_weights = estimate_swarm_GMM(conbinedmeans_list, conbinedcovs_list,
                                                                          robots_positions)
        print(f'estimate_swarm_GMM+APF{time.time() - start}')
        
        current_goal_means, current_goal_covs, current_goal_weights = next_means, next_covs, next_weights
        _, _, WG = calWGMetric_speedUp(current_means, current_covs, current_weights, current_goal_means,
                                       current_goal_covs, current_goal_weights)
        print(f'estimate_swarm_GMM+APF+WG{time.time() - start}')
        diff_gmm_est_targ.append(WG)
        '''
        dif_J_rate = J_rate_pre - J_rate
        n = n + 1
        J_rate_pre = J_rate
    numTry = n
    return robots_positions, robots_positions_list, J_rate, numTry, diff_gmm_est_targ



def agentControl_APF(next_means, next_covs, next_weights, robots_positions, esdf_map, MaxNumTry):

    xa, ya, za = esdf_map.origin
    xb = xa + esdf_map.dims[0] * esdf_map.resolution
    yb = ya + esdf_map.dims[1] * esdf_map.resolution
    zb = za + esdf_map.dims[2] * esdf_map.resolution
    # Robot motion constraints
    max_Velocity = APF_MAX_VELOCITY_BASE
    r_repulsion_sensor = 0.35           # 机间开始明显斥力的距离 [m]，略增大使力变化更平滑
    r_repulsion_obstacle = 0.28         # ESDF 斥力作用带 [m]
    dimW = int(3)
    numAgent = robots_positions.shape[0]   
    minDistance = 1e-8                  
    gamma = APF_GAMMA_INTER_AGENT
    sigma_k = np.eye(dimW) * APF_SIGMA_K_DIAG
    Rdiameter = 0.35                    # 机间安全等效直径 [m]（原 0.012 过小，斥力几乎始终极强）
    Rradius = 0.5 * Rdiameter
    if MaxNumTry != 1:
        max_Velocity = max_Velocity / MaxNumTry * 2.0
    #numComponent = len(next_means)
    #dU = robots_positions.shape          
    #GM_gmm = np.zeros(robots_positions.shape[0])
    dU_sensor_gmm = np.zeros((numAgent, dimW))          # gradient of attraction force
    U_sensor_gmm = np.zeros(numAgent)                   # scalar potentials
    Mu = next_means
    Sigma = next_covs
    Weight = next_weights  

    '''compute attractive force(gradient of GMM potential)
       this force can make robot to move towards the peak direction of Gaussian distribution
    '''
    for l in range(len(next_means)):
        sigma = Sigma[l] + sigma_k                      # 调整后的协方差矩阵
        mu = np.array(Mu[l])                            # 均值（三维空间坐标）
        weight = Weight[l]                              # 权重
        GM_gmm = multivariate_normal.pdf(robots_positions, mean=mu, cov=sigma)    # 每个机器人在第l个高斯分布下概率密度（数值越大说明更靠近该分布中心）
        Diff_sensor_gmm = robots_positions - mu         # 机器人位置与高斯分布中心的差
        dU_sensor_gmm = dU_sensor_gmm + weight * np.hstack([GM_gmm[:, np.newaxis], GM_gmm[:, np.newaxis], GM_gmm[:, np.newaxis]]) * (
                Diff_sensor_gmm @ np.linalg.inv(sigma))
        U_sensor_gmm = U_sensor_gmm + weight * GM_gmm
    dU_sensor_gmm = (1 / numAgent) * dU_sensor_gmm * APF_ATTRACT_GAIN
    U_sensor_gmm = -np.mean(U_sensor_gmm)

    '''
    compute repulsion force1 between agents(avoid collision)   
    method: use Gaussian kernal to smooth the change of repulsion force
    '''
    Diff_sensor = np.zeros((numAgent ** 2, dimW))                              # matrix store pairwise differences
    for n in range(dimW):
        SensorPos_vector = np.expand_dims(robots_positions[:, n], axis=1)      # extract the nth column of robot_positions(num_Agent,dimw)
        Diff_sensor_matrix = SensorPos_vector - np.transpose(SensorPos_vector) # This broadcasts the column and row vectors to create a matrix of pairwise differences.
        Diff_sensor[:, n] = Diff_sensor_matrix.flatten(order='F')              # flatten the matrix to a column array(numagent**2,1)
    mu = np.zeros(dimW)  
    sigma = 2 * sigma_k
    GM_sensor_vector = np.expand_dims(multivariate_normal.pdf(Diff_sensor, mean=mu, cov=sigma), axis=1) # 算出每对机器人之间距离的在标准三维正态分布下的pdf值作为势能
    dU_sensor_vector = (GM_sensor_vector * (Diff_sensor @ np.linalg.inv(sigma))).flatten(order='F')     # 对高斯势场求梯度
    dU_sensor_matrix = dU_sensor_vector.reshape((dimW, numAgent, numAgent)).transpose(0, 2, 1)
    # 高斯机间势的梯度（仅进入标量 U 的 gamma 项；位移主方向见下式 dU 中的反比距离项）
    dU_sensor = (
        -np.sum(dU_sensor_matrix, axis=1).T / (numAgent**2) * APF_AGENT_REPULSE_GAIN
    )
    U_sensor = 1 / 2 * np.mean(GM_sensor_vector)

    # compute repulsion force2 between agents(avoid collision)  method: using inverse distance to ensure a strict min distance
    Dist_sensor = cdist(robots_positions, robots_positions)
    Dist_sensor = Dist_sensor.flatten(order='F')
    dd = Rdiameter * np.ones(Dist_sensor.shape[0])
    Dist_sensor = Dist_sensor - dd
    dU_repulsion_sensor_vector = np.zeros((numAgent ** 2, dimW))
    index_other_near = np.where((Dist_sensor > 0) * (Dist_sensor <= r_repulsion_sensor))[0]
    if not index_other_near.size == 0:
        Dist_sensor_repulsion = Dist_sensor[index_other_near]
        Diff_sensor_repulsion = Diff_sensor[index_other_near, :]
        dU_repulsion_sensor_other_near = (1 / Dist_sensor_repulsion - 1 / r_repulsion_sensor) * (
                Dist_sensor_repulsion ** (-3))               # dU_repulsion_sensor_other_near = (1/D-1/r)*(D)^-3
        dU_repulsion_sensor_other_near = np.tile(np.expand_dims(dU_repulsion_sensor_other_near, axis=1),
                                                 (1, dimW)) * Diff_sensor_repulsion
        dU_repulsion_sensor_vector[index_other_near, :] = dU_repulsion_sensor_other_near
    dU_repulsion_sensor_vector = dU_repulsion_sensor_vector.flatten(order='F')
    dU_repulsion_sensor_matrix = np.reshape(dU_repulsion_sensor_vector, (dimW, numAgent, numAgent)).transpose(0, 2, 1)
    dU_repulsion_sensor = -np.sum(dU_repulsion_sensor_matrix, 2).T * APF_AGENT_REPULSE_GAIN


    # dU_repulsion_obstacle
    '''
    基于已知障碍物下的避障排斥力
    obstacle_vertices = [obstacle.vertices for obstacle in obstacle_manager.obstacles]
    Obstacles = [trimesh.convex.convex_hull(vertices) for vertices in obstacle_vertices]
    numObstacles = len(Obstacles)
    numAgent = len(robots_positions)
    dimW = 3 
    Dist_sensor_obstacle = np.zeros((numAgent, numObstacles))
    nj = np.zeros((numAgent, numObstacles, dimW))  # store 3-D normal_vector
    for i in range(numAgent):
        for j in range(numObstacles):
            nj[i, j] = normal_vector_SDF_3d(robots_positions[i], obstacle_vertices[j])
            Dist_sensor_obstacle[i, j], _ = sd_3d(robots_positions[i], obstacle_vertices[j])
    rr = Rradius * np.ones(Dist_sensor_obstacle.size)
    Dist_sensor_obstacle_flat = Dist_sensor_obstacle.flatten(order='F') - rr
    index_obstacle_near = np.where((Dist_sensor_obstacle_flat > 0) & (Dist_sensor_obstacle_flat <= r_repulsion_obstacle))[0]

    dU_repulsion_sensor_obstacle = np.zeros((numAgent, dimW))

    if index_obstacle_near.size > 0:
        Dist_sensor_obstacle_repulsion = Dist_sensor_obstacle_flat[index_obstacle_near]
        nj_flat = nj.reshape(-1, dimW, order='F')  # shape (numAgent*numObstacles, 2)
        nj_filtered = nj_flat[index_obstacle_near]  # shape (n_near, 2)
        epsilon = 1e-10  # escape from dividing 0
        safe_dist = Dist_sensor_obstacle_repulsion + epsilon
        scale_factor = (1 / safe_dist - 1 / r_repulsion_obstacle) * (safe_dist)** (-3)
        dU_repulsion_sensor_obstacle_other_near = scale_factor[:, np.newaxis] * nj_filtered  # shape (n_near, dimW)
    
        agent_indices = index_obstacle_near % numAgent  # restore Agent index（key step）  k = i + j * numAgent
        # repulsion to agent i is the sum of every obstacle 
        np.add.at(dU_repulsion_sensor_obstacle, (agent_indices, slice(None)), dU_repulsion_sensor_obstacle_other_near)
    '''
    num_agents = len(robots_positions)
    dim = 3
    dU_repulsion_sensor_obstacle = np.zeros((num_agents, dim))
    # 获取所有智能体的ESDF信息
    esdf_distances = np.zeros(num_agents)
    esdf_gradients = np.zeros((num_agents, dim))
    
    for i, pos in enumerate(robots_positions):
        d = esdf_map.get_esdf(pos)
        esdf_distances[i] = d - Rradius
        grad = esdf_map.compute_gradient(pos)
        esdf_gradients[i] = grad if grad is not None else np.zeros(dim)  
    
    # 计算需要施加斥力的智能体索引
    in_repulsion_zone = (esdf_distances > 0) & (esdf_distances <= r_repulsion_obstacle)
    affected_agents = np.where(in_repulsion_zone)[0]
    
    if affected_agents.size > 0:
        # 计算斥力强度系数（距离下限防 1/d^4 爆炸）
        safe_dists = np.maximum(esdf_distances[affected_agents], APF_ESDF_SAFE_DIST_MIN)
        epsilon = 1e-10
        scale_factors = (1/(safe_dists + epsilon) - 1/r_repulsion_obstacle) / (safe_dists**3 + epsilon)
        
        # 斥力沿 −∇d，使 −(dU 总和) 在障碍侧指向自由空间（与吸引项一致的下坡约定）
        repulsion_vectors = -scale_factors[:, np.newaxis] * esdf_gradients[affected_agents]
        
        # 累加到总斥力矩阵
        np.add.at(dU_repulsion_sensor_obstacle, (affected_agents, slice(None)), repulsion_vectors)
    dU_repulsion_sensor_obstacle *= APF_ESDF_REPULSE_GAIN

    # 边界斥力
    Dist_sensor_boundary_Left = robots_positions[:, 0] - xa + minDistance - Rradius
    Dist_sensor_boundary_Right = xb - robots_positions[:, 0] + minDistance - Rradius
    Dist_sensor_boundary_Top = yb - robots_positions[:, 1] + minDistance - Rradius
    Dist_sensor_boundary_Bottom = robots_positions[:, 1] - ya + minDistance - Rradius
    Dist_sensor_boundary_Front = robots_positions[:, 2] - za + minDistance - Rradius
    Dist_sensor_boundary_Back = zb - robots_positions[:, 2] + minDistance - Rradius
    Diff_sensor_boundary_Left   = np.array([ Dist_sensor_boundary_Left,  np.zeros(numAgent), np.zeros(numAgent)]).T
    Diff_sensor_boundary_Right  = np.array([-Dist_sensor_boundary_Right, np.zeros(numAgent), np.zeros(numAgent)]).T
    Diff_sensor_boundary_Top    = np.array([ np.zeros(numAgent), -Dist_sensor_boundary_Top, np.zeros(numAgent)]).T
    Diff_sensor_boundary_Bottom = np.array([ np.zeros(numAgent),  Dist_sensor_boundary_Bottom, np.zeros(numAgent)]).T
    Diff_sensor_boundary_Front  = np.array([ np.zeros(numAgent), np.zeros(numAgent),  Dist_sensor_boundary_Front]).T
    Diff_sensor_boundary_Back   = np.array([ np.zeros(numAgent), np.zeros(numAgent), -Dist_sensor_boundary_Back]).T

    idx_Left = np.where(Dist_sensor_boundary_Left <= r_repulsion_obstacle)[0]
    idx_Right = np.where(Dist_sensor_boundary_Right <= r_repulsion_obstacle)[0]
    idx_Top = np.where(Dist_sensor_boundary_Top <= r_repulsion_obstacle)[0]
    idx_Bottom = np.where(Dist_sensor_boundary_Bottom <= r_repulsion_obstacle)[0]
    idx_Front = np.where(Dist_sensor_boundary_Front <= r_repulsion_obstacle)[0]
    idx_Back = np.where(Dist_sensor_boundary_Back <= r_repulsion_obstacle)[0]
    dU_repulsion_Left = np.zeros((numAgent, dimW))
    dU_repulsion_Right = np.zeros((numAgent, dimW))
    dU_repulsion_Top = np.zeros((numAgent, dimW))
    dU_repulsion_Bottom = np.zeros((numAgent, dimW))
    dU_repulsion_Front = np.zeros((numAgent, dimW))
    dU_repulsion_Back = np.zeros((numAgent, dimW))

    '''
    if not idx_Left.shape[0] == 0:
        dU_repulsion_Left[idx_Left, :] = -(1 / Dist_sensor_boundary_Left[idx_Left] - 1 / r_repulsion_obstacle)[:, np.newaxis] * (
                        Dist_sensor_boundary_Left[idx_Left] ** (-3))[:, np.newaxis] * Diff_sensor_boundary_Left[idx_Left, :]
    if not idx_Right.shape[0] == 0:
        dU_repulsion_Right[idx_Right, :] = -(1 / Dist_sensor_boundary_Right[idx_Right] - 1 / r_repulsion_obstacle)[:, np.newaxis] * (
                Dist_sensor_boundary_Right[idx_Right] ** (-3))[:, np.newaxis] * Diff_sensor_boundary_Right[idx_Right, :]
    if not idx_Top.shape[0] == 0:
        dU_repulsion_Top[idx_Top, :] = -(1. / Dist_sensor_boundary_Top[idx_Top] - 1 / r_repulsion_obstacle)[:, np.newaxis] * (
                Dist_sensor_boundary_Top[idx_Top] ** (-3))[:, np.newaxis] * Diff_sensor_boundary_Top[idx_Top, :]
    if not idx_Bottom.shape[0] == 0:
        dU_repulsion_Bottom[idx_Bottom, :] = -(
                1 / Dist_sensor_boundary_Bottom[idx_Bottom] - 1 / r_repulsion_obstacle)[:, np.newaxis] * (
                                                     Dist_sensor_boundary_Bottom[idx_Bottom] ** (
                                                 -3))[:, np.newaxis] * Diff_sensor_boundary_Bottom[idx_Bottom, :]
    if not idx_Front.shape[0] == 0:
        dU_repulsion_Front[idx_Front, :] = -(1 / Dist_sensor_boundary_Front[idx_Front] - 1 / r_repulsion_obstacle)[:, np.newaxis] * (
                Dist_sensor_boundary_Front[idx_Front] ** (-3))[:, np.newaxis] * Diff_sensor_boundary_Front[idx_Front, :]
    if not idx_Back.shape[0] == 0:
        dU_repulsion_Back[idx_Back, :] = -(1 / Dist_sensor_boundary_Back[idx_Back] - 1 / r_repulsion_obstacle)[:, np.newaxis] * (
                Dist_sensor_boundary_Back[idx_Back] ** (-3))[:, np.newaxis] * Diff_sensor_boundary_Back[idx_Back, :]

    dU_repulsion_sensor_boundary = dU_repulsion_Left + dU_repulsion_Right + dU_repulsion_Top + dU_repulsion_Bottom + dU_repulsion_Front + dU_repulsion_Back
    '''
    # 定义力的最大绝对值限制
    force_limit = 300000.0  # 根据实际情况调整这个值

    if not idx_Left.shape[0] == 0:
        dU_repulsion_Left[idx_Left, :] = -(1 / Dist_sensor_boundary_Left[idx_Left] - 1 / r_repulsion_obstacle)[:, np.newaxis] * (
                        Dist_sensor_boundary_Left[idx_Left] ** (-3))[:, np.newaxis] * Diff_sensor_boundary_Left[idx_Left, :]
        # 对Left方向的力进行限幅
        dU_repulsion_Left = np.clip(dU_repulsion_Left, -force_limit, force_limit)

    if not idx_Right.shape[0] == 0:
        dU_repulsion_Right[idx_Right, :] = -(1 / Dist_sensor_boundary_Right[idx_Right] - 1 / r_repulsion_obstacle)[:, np.newaxis] * (
                Dist_sensor_boundary_Right[idx_Right] ** (-3))[:, np.newaxis] * Diff_sensor_boundary_Right[idx_Right, :]
        # 对Right方向的力进行限幅
        dU_repulsion_Right = np.clip(dU_repulsion_Right, -force_limit, force_limit)

    if not idx_Top.shape[0] == 0:
        dU_repulsion_Top[idx_Top, :] = -(1. / Dist_sensor_boundary_Top[idx_Top] - 1 / r_repulsion_obstacle)[:, np.newaxis] * (
                Dist_sensor_boundary_Top[idx_Top] ** (-3))[:, np.newaxis] * Diff_sensor_boundary_Top[idx_Top, :]
        # 对Top方向的力进行限幅
        dU_repulsion_Top = np.clip(dU_repulsion_Top, -force_limit, force_limit)

    if not idx_Bottom.shape[0] == 0:
        dU_repulsion_Bottom[idx_Bottom, :] = -(
                1 / Dist_sensor_boundary_Bottom[idx_Bottom] - 1 / r_repulsion_obstacle)[:, np.newaxis] * (
                                                    Dist_sensor_boundary_Bottom[idx_Bottom] ** (
                                                -3))[:, np.newaxis] * Diff_sensor_boundary_Bottom[idx_Bottom, :]
        # 对Bottom方向的力进行限幅
        dU_repulsion_Bottom = np.clip(dU_repulsion_Bottom, -force_limit, force_limit)

    if not idx_Front.shape[0] == 0:
        dU_repulsion_Front[idx_Front, :] = -(1 / Dist_sensor_boundary_Front[idx_Front] - 1 / r_repulsion_obstacle)[:, np.newaxis] * (
                Dist_sensor_boundary_Front[idx_Front] ** (-3))[:, np.newaxis] * Diff_sensor_boundary_Front[idx_Front, :]
        # 对Front方向的力进行限幅
        dU_repulsion_Front = np.clip(dU_repulsion_Front, -force_limit, force_limit)

    if not idx_Back.shape[0] == 0:
        dU_repulsion_Back[idx_Back, :] = -(1 / Dist_sensor_boundary_Back[idx_Back] - 1 / r_repulsion_obstacle)[:, np.newaxis] * (
                Dist_sensor_boundary_Back[idx_Back] ** (-3))[:, np.newaxis] * Diff_sensor_boundary_Back[idx_Back, :]
        # 对Back方向的力进行限幅
        dU_repulsion_Back = np.clip(dU_repulsion_Back, -force_limit, force_limit)

    dU_repulsion_sensor_boundary = (
        dU_repulsion_Left
        + dU_repulsion_Right
        + dU_repulsion_Top
        + dU_repulsion_Bottom
        + dU_repulsion_Front
        + dU_repulsion_Back
    ) * APF_BOUNDARY_REPULSE_GAIN
    # 合成方向：先分别单位化吸引与「各类斥力之和」，再混合。避免斥力模长远大于吸引时，
    # 「矢量相加再归一」把目标方向几乎消掉 → 贴障滑移/局部极小/过不了中间有障碍的路径。
    att = dU_sensor_gmm
    rep = (
        dU_repulsion_sensor
        + dU_repulsion_sensor_obstacle
        + dU_repulsion_sensor_boundary
    )
    att_n = np.linalg.norm(att, axis=1, keepdims=True)
    rep_n = np.linalg.norm(rep, axis=1, keepdims=True)
    att_u = np.divide(att, att_n, out=np.zeros_like(att), where=(att_n > 1e-10))
    rep_u = np.divide(rep, rep_n, out=np.zeros_like(rep), where=(rep_n > 1e-10))
    ba = float(APF_ATT_REP_UNIT_BLEND)
    dU = ba * att_u + (1.0 - ba) * rep_u
    comb_n = np.linalg.norm(dU, axis=1, keepdims=True)
    dU = np.divide(dU, np.maximum(comb_n, 1e-10))
    # 吸引与斥力近乎对顶时 comb≈0，优先保持指向目标的 att_u
    oppose = comb_n.ravel() < 1e-8
    if np.any(oppose):
        for j in np.flatnonzero(oppose):
            if att_n[j, 0] > 1e-10:
                dU[j, :] = att_u[j, :]
            elif rep_n[j, 0] > 1e-10:
                dU[j, :] = rep_u[j, :]
    U = U_sensor_gmm + gamma * U_sensor
    # guarantee free from collision
    # guarantee free from collision
    dists = esdf_map.get_esdf(robots_positions)   # robots_positions shape (N, 3)
    if np.any(dists <= 0):
        print('There are some sensors in the obstacle areas!!!')

    # 1. 预测一步：先按最大速度走
    SensorPos_next = robots_positions - dU * max_Velocity          # shape (N,3)
    # ------------------------------------------------------------------
    # 2. 检测碰撞（ESDF ≤ 0 表示碰到障碍物或穿透）
    dists        = esdf_map.get_esdf(SensorPos_next)               # shape (N,)
    indexSensor  = np.where(dists <= 0)[0]                         # 整数索引数组
    k = 1

    # ------------------------------------------------------------------
    # 3. 自适应减速：最多尝试 3 次，每次速度减半
    while k <= 3 and not indexSensor.shape[0] == 0:
        Velocity = max_Velocity * (1/2) ** k  # NEW: 防止极端情况下全零位移   
        SensorPos_next[indexSensor] = (
            robots_positions[indexSensor] - dU[indexSensor] * Velocity
        )
        # 重新检测
        dists       = esdf_map.get_esdf(SensorPos_next)
        indexSensor = np.where(dists <= 0)[0]
        k += 1

    # 第 4 次仍然碰撞 → 先退回原点，再试沿「纯吸引」方向极小步，减少卡死在障碍壳上
    if indexSensor.size > 0:
        stuck = np.asarray(indexSensor, dtype=int).ravel()
        SensorPos_next[stuck] = robots_positions[stuck]
        for ii in stuck:
            ga = dU_sensor_gmm[ii, :]
            gn = float(np.linalg.norm(ga))
            if gn < 1e-12:
                continue
            gu = ga / gn
            for frac in (0.45, 0.28, 0.16, 0.09, 0.05):
                trial = robots_positions[ii, :] - gu * (max_Velocity * frac)
                d_try = esdf_map.get_esdf(trial)
                if isinstance(d_try, np.ndarray):
                    d_try = float(np.asarray(d_try).ravel()[0])
                else:
                    d_try = float(d_try)
                if d_try > 0:
                    SensorPos_next[ii, :] = trial
                    break

    # ------------------------------------------------------------------
    # 4. 边界约束（修正 z 维下标写错的问题）
    SensorPos_next[:, 0] = np.minimum(
        np.maximum(SensorPos_next[:, 0], xa + minDistance + 0.5 * Rdiameter),
        xb - minDistance - 0.5 * Rdiameter
    )
    SensorPos_next[:, 1] = np.minimum(
        np.maximum(SensorPos_next[:, 1], ya + minDistance + 0.5 * Rdiameter),
        yb - minDistance - 0.5 * Rdiameter
    )
    SensorPos_next[:, 2] = np.minimum(                           
        np.maximum(SensorPos_next[:, 2], za + minDistance + 0.5 * Rdiameter),
        zb - minDistance - 0.5 * Rdiameter
    )

    # ------------------------------------------------------------------
    # 5. 后续势能与协同项的计算
    U_sensor_next_gmm = np.zeros(numAgent)
    for l in range(len(next_means)):
        sigma  = Sigma[l] + sigma_k
        mu     = Mu[l]
        weight = Weight[l]
        GM_gmm = multivariate_normal.pdf(SensorPos_next, mean=mu, cov=sigma)
        U_sensor_next_gmm += weight * GM_gmm
    U_sensor_next_gmm = -np.mean(U_sensor_next_gmm)

    Diff_sensor_next = np.zeros((numAgent ** 2, dimW))
    for n in range(dimW):
        SensorPos_next_vector = SensorPos_next[:, n]
        Diff_sensor_next_matrix = np.expand_dims(SensorPos_next_vector, axis=1) - np.expand_dims(SensorPos_next_vector,
                                                                                                 axis=1).T
        Diff_sensor_next[:, n] = Diff_sensor_next_matrix.flatten(order='F')

    mu0   = np.zeros(dimW)
    sigma = 2 * sigma_k
    GM_sensor_next_vector = multivariate_normal.pdf(
        Diff_sensor_next, mean=mu0, cov=sigma
    )
    U_sensor_next = 0.5 * np.mean(GM_sensor_next_vector)

    U_next = U_sensor_next_gmm + gamma * U_sensor_next
    J_rate              = U - U_next
    robots_positions_next = SensorPos_next
    # ------------------------------------------------------------------
    return robots_positions_next, J_rate, U, U_next

'''
def infree(points, obstacle_manager):  # if a collision happened, return False
    numPoints = points.shape[0]
    InFreeSpaceFlag = True
    TF = []
    for n in range(numPoints):
        InFreeSpaceFlag = not obstacle_manager.is_colliding(points[n][0], points[n][1], points[n][2])
        TF.append(InFreeSpaceFlag)
    return TF
'''
'''
def Boundary_points_shape(dx, dy, xa, ya, xb, yb, obstacle_vertices):
    numGridX = int((2 * xb) / dx + 1)
    numGridY = int((2 * yb) / dy + 1)
    xGrid = np.linspace(-xb / 2, 1.5 * xb, numGridX)
    yGrid = np.linspace(-yb / 2, 1.5 * yb, numGridY)
    gridX, gridY = np.meshgrid(xGrid, yGrid)
    gridXY = np.column_stack((gridX.flatten(order='F'), gridY.flatten(order='F')))
    # idx_true_HM = np.where((xa <= gridXY[:, 0]) & (gridXY[:, 0] <= xb) &
    #                       (ya <= gridXY[:, 1]) & (gridXY[:, 1] <= yb))[0]
    # HM = HM[idx_true_HM]
    numGridX = int((xb - xa) / dx + 1)
    numGridY = int((yb - ya) / dy + 1)
    xGrid = np.linspace(xa, xb, numGridX)
    yGrid = np.linspace(ya, yb, numGridY)
    gridX, gridY = np.meshgrid(xGrid, yGrid)
    gridX, gridY = np.round(gridX, decimals=1), np.round(gridY, decimals=1)
    gridXY = np.column_stack((gridX.flatten(order='F'), gridY.flatten(order='F')))
    # HM = HM.reshape((gridX.shape[1], gridX.shape[0])).T
    # contours = find_contours(HM, level=0.5)
    Obstacles = [ShapelyPolygon(vertices) for vertices in obstacle_vertices]

    BoundaryPoints = []
    for i in range(len(Obstacles)):
        obstacle = Obstacles[i]
        idx = np.ravel_multi_index((obstacle[:, 0], obstacle[:, 1]), (gridXY.shape[0],gridXY.shape[1]), order='F')
        BoundaryPoints.append(gridXY[idx, :])
    BoundaryPoints = np.vstack(BoundaryPoints)
    return BoundaryPoints
'''

def estimate_swarm_GMM(conbinedmeans_list, conbinedcovs_list, robots_positions):
    N = robots_positions.shape[0]
    Mu = np.array(conbinedmeans_list)
    distance = cdist(robots_positions, Mu)
    sorted_indices = np.argsort(distance, axis=1)
    id_nearest_gaussians = sorted_indices[:, :4]
    P = 3
    pdf_values = np.zeros((N, P))
    for k in range(P):
        idx = id_nearest_gaussians[:, k]
        mu_k = Mu[idx, :]
        cov_k = np.stack([conbinedcovs_list[i] for i in idx], axis=0)
        for i in range(robots_positions.shape[0]):
            pdf_values[i, k] = multivariate_normal.pdf(robots_positions[i], mean=mu_k[i], cov=cov_k[i])
    max_pdf_local_indices = np.argmax(pdf_values, axis=1)
    max_pdf_indices = id_nearest_gaussians[np.arange(N), max_pdf_local_indices]
    id_est, ic = np.unique(max_pdf_indices, return_inverse=True)
    GC_count = np.bincount(ic)

    NsigmaT = [conbinedcovs_list[index] for index in id_est]
    NmeanT = Mu[id_est].tolist()
    NwT = (GC_count / N).tolist()
    return NmeanT, NsigmaT, NwT

def estimate_swarm_GMM_3D(conbinedmeans_list, conbinedcovs_list, robots_positions):
    """
    三维空间下的高斯分布关联处理
    ----------------------------------------------
    robots_positions     : (N,3) ndarray
    conbinedmeans_list   : list  [ [x,y,z], ... ]
    conbinedcovs_list    : list  [ 3×3 ndarray, ... ]
    ----------------------------------------------
    返回 NmeanT, NsigmaT, NwT
    """
    # ---------- 输入校验 ----------
    assert robots_positions.ndim == 2 and robots_positions.shape[1] == 3, "机器人坐标必须是 (N,3)"
    assert all(len(m) == 3 for m in conbinedmeans_list),               "均值必须是三维向量"
    assert all(np.asarray(c).shape == (3, 3) for c in conbinedcovs_list), "协方差必须 3×3"

    # ---------- 预处理 ----------
    Mu = np.asarray(conbinedmeans_list, dtype=float)   # (M,3)
    M  = Mu.shape[0]
    if M == 0:                                         # 没有高斯分布
        return [], [], []

    N = robots_positions.shape[0]
    P = min(4, M)                                      # 最近邻个数

    # ---------- 距离与最近邻 ----------
    dist   = cdist(robots_positions, Mu)               # (N,M)
    nn_idx = np.argsort(dist, axis=1)[:, :P]           # (N,P)

    # ---------- 概率密度 ----------
    pdf = np.zeros((N, P))
    for k in range(P):
        idx     = nn_idx[:, k]                         # (N,)
        mu_k    = Mu[idx]                              # (N,3)
        cov_k   = np.stack([conbinedcovs_list[i] for i in idx])
        for i in range(N):
            try:
                pdf[i, k] = multivariate_normal.pdf(robots_positions[i],
                                                    mean=mu_k[i],
                                                    cov=cov_k[i])
            except Exception:
                pdf[i, k] = 0.0                        # 奇异协方差等情况

    # ---------- 取最大概率对应的高斯 ----------
    winner_local = np.argmax(pdf, axis=1)              # (N,)
    winner_idx   = nn_idx[np.arange(N), winner_local]  # (N,)

    # ---------- 统计权重 ----------
    unique_idx, inverse = np.unique(winner_idx, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique_idx))          # 选择次数
    weights = counts / N                                              # 归一化

    # ---------- 输出 ----------
    NmeanT  = Mu[unique_idx].tolist()
    NsigmaT = [conbinedcovs_list[i] for i in unique_idx]
    NwT     = weights.tolist()

    return NmeanT, NsigmaT, NwT

def fit_swarm_GMM(robots_positions, num_weights):
    gmm = GaussianMixture(n_components=num_weights, covariance_type='full')
    gmm.fit(robots_positions)
    weights = gmm.weights_.tolist()
    means = gmm.means_.tolist()
    covs = gmm.covariances_.tolist()
    return means, covs, weights

def Projection_Trajectory_Point(current_points, points_list):  # return segmented trajectories
    '''
    :param current_points: n * 2 array
    :param points_list:  n * (length(trajectory) * 2 array)  list
    '''
    def projection_and_next_point(point, segments):
        line = LineString(segments)
        p = Point(point)
        proj_point = line.interpolate(line.project(p))
        for i in range(len(segments) - 1):
            segment = LineString([segments[i], segments[i + 1]])
            if segment.distance(proj_point) < 1e-8:
                next_point = segments[i + 1]
                break
        else:
            next_point = segments[-1]
        return proj_point, next_point

    trajectories = []
    num_robots = len(points_list)
    for i in range(num_robots):
        line = []
        num_points = len(points_list[i])
        for j in range(num_points):
            line.append(points_list[i][j].reshape(2))
        _, next_point = projection_and_next_point(current_points[i], np.array(line))
        index = np.where((next_point == np.array(line)).all(axis=1))[0]
        tracjectory = points_list[i][int(index[0]):]
        trajectories.append(tracjectory)
    return trajectories

import numpy as np
from scipy.spatial.distance import cdist

def get_3d_trajectories(current_points, points_list):
    """
    三维轨迹分割函数（无投影）
    
    参数：
    current_points : num_robots x 3 数组 - 当前三维坐标
    points_list : 列表，每个元素是 n x 3 数组 - 三维轨迹点序列
    
    返回：
    列表，每个元素是从最近点开始的后续三维轨迹
    """
    def find_closest_index(query_point, trajectory):
        """找到轨迹中离当前点最近的点索引"""
        # 计算所有轨迹点到查询点的距离
        distances = cdist(trajectory, np.array([query_point]))
        return np.argmin(distances)

    trajectories = []
    num_robots = len(points_list)
    
    for i in range(num_robots):
        # 转换轨迹为三维数组 (N x 3)
        traj_3d = np.array(points_list[i]).reshape(-1, 3)
        
        # 找到当前点在轨迹中的最近索引
        closest_idx = find_closest_index(current_points[i], traj_3d)
        
        # 截取后续轨迹 (包含当前最近点)
        segmented_traj = traj_3d[closest_idx:]
        
        trajectories.append(segmented_traj)
        
    return trajectories





