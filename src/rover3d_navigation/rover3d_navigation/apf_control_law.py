import time

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal

# ---------------------------------------------------------------------------
# APF 调参
# ---------------------------------------------------------------------------
APF_SIGMA_K_DIAG = 0.65
APF_ATTRACT_GAIN = 1.8
APF_AGENT_REPULSE_GAIN = 0.18
APF_ESDF_REPULSE_GAIN = 0.24
APF_ESDF_SAFE_DIST_MIN = 0.08
APF_ATT_REP_UNIT_BLEND = 0.76
APF_BOUNDARY_REPULSE_GAIN = 0.24
APF_GAMMA_INTER_AGENT = 0.14
APF_MAX_VELOCITY_BASE = 0.48
APF_OBSTACLE_WARN_PERIOD_SEC = 2.0
_last_obstacle_warning_time = 0.0


def APF(next_means, next_covs, next_weights, robots_positions, esdf_map, MaxNumTry=10):
    n = 1
    J_rate = float("inf")
    J_rate_pre = float("inf")
    diff_gmm_est_targ = []
    robots_positions_list = []
    while n <= MaxNumTry and J_rate > 1e-6:
        _ = time.time()
        robots_positions, J_rate, _, _ = agentControl_APF(
            next_means, next_covs, next_weights, robots_positions, esdf_map, MaxNumTry
        )
        robots_positions_list.append(robots_positions)
        _ = J_rate_pre - J_rate
        n = n + 1
        J_rate_pre = J_rate
    numTry = n
    return robots_positions, robots_positions_list, J_rate, numTry, diff_gmm_est_targ


def agentControl_APF(next_means, next_covs, next_weights, robots_positions, esdf_map, MaxNumTry):
    xa, ya, za = esdf_map.origin
    xb = xa + esdf_map.dims[0] * esdf_map.resolution
    yb = ya + esdf_map.dims[1] * esdf_map.resolution
    zb = za + esdf_map.dims[2] * esdf_map.resolution
    max_Velocity = APF_MAX_VELOCITY_BASE
    r_repulsion_sensor = 0.35
    r_repulsion_obstacle = 0.28
    dimW = 3
    numAgent = robots_positions.shape[0]
    minDistance = 1e-8
    gamma = APF_GAMMA_INTER_AGENT
    sigma_k = np.eye(dimW) * APF_SIGMA_K_DIAG
    Rdiameter = 0.35
    Rradius = 0.5 * Rdiameter
    if MaxNumTry != 1:
        max_Velocity = max_Velocity / MaxNumTry * 2.0

    dU_sensor_gmm = np.zeros((numAgent, dimW))
    U_sensor_gmm = np.zeros(numAgent)
    Mu = next_means
    Sigma = next_covs
    Weight = next_weights

    for l in range(len(next_means)):
        sigma = Sigma[l] + sigma_k
        mu = np.array(Mu[l])
        weight = Weight[l]
        GM_gmm = multivariate_normal.pdf(robots_positions, mean=mu, cov=sigma)
        Diff_sensor_gmm = robots_positions - mu
        dU_sensor_gmm = dU_sensor_gmm + weight * np.hstack(
            [GM_gmm[:, np.newaxis], GM_gmm[:, np.newaxis], GM_gmm[:, np.newaxis]]
        ) * (Diff_sensor_gmm @ np.linalg.inv(sigma))
        U_sensor_gmm = U_sensor_gmm + weight * GM_gmm
    dU_sensor_gmm = (1 / numAgent) * dU_sensor_gmm * APF_ATTRACT_GAIN
    U_sensor_gmm = -np.mean(U_sensor_gmm)

    Diff_sensor = np.zeros((numAgent**2, dimW))
    for n in range(dimW):
        SensorPos_vector = np.expand_dims(robots_positions[:, n], axis=1)
        Diff_sensor_matrix = SensorPos_vector - np.transpose(SensorPos_vector)
        Diff_sensor[:, n] = Diff_sensor_matrix.flatten(order="F")
    mu = np.zeros(dimW)
    sigma = 2 * sigma_k
    GM_sensor_vector = np.expand_dims(multivariate_normal.pdf(Diff_sensor, mean=mu, cov=sigma), axis=1)
    dU_sensor_vector = (GM_sensor_vector * (Diff_sensor @ np.linalg.inv(sigma))).flatten(order="F")
    dU_sensor_matrix = dU_sensor_vector.reshape((dimW, numAgent, numAgent)).transpose(0, 2, 1)
    dU_sensor = -np.sum(dU_sensor_matrix, axis=1).T / (numAgent**2) * APF_AGENT_REPULSE_GAIN
    U_sensor = 0.5 * np.mean(GM_sensor_vector)

    Dist_sensor = cdist(robots_positions, robots_positions)
    Dist_sensor = Dist_sensor.flatten(order="F")
    dd = Rdiameter * np.ones(Dist_sensor.shape[0])
    Dist_sensor = Dist_sensor - dd
    dU_repulsion_sensor_vector = np.zeros((numAgent**2, dimW))
    index_other_near = np.where((Dist_sensor > 0) * (Dist_sensor <= r_repulsion_sensor))[0]
    if not index_other_near.size == 0:
        Dist_sensor_repulsion = Dist_sensor[index_other_near]
        Diff_sensor_repulsion = Diff_sensor[index_other_near, :]
        dU_repulsion_sensor_other_near = (1 / Dist_sensor_repulsion - 1 / r_repulsion_sensor) * (
            Dist_sensor_repulsion ** (-3)
        )
        dU_repulsion_sensor_other_near = np.tile(
            np.expand_dims(dU_repulsion_sensor_other_near, axis=1), (1, dimW)
        ) * Diff_sensor_repulsion
        dU_repulsion_sensor_vector[index_other_near, :] = dU_repulsion_sensor_other_near
    dU_repulsion_sensor_vector = dU_repulsion_sensor_vector.flatten(order="F")
    dU_repulsion_sensor_matrix = np.reshape(dU_repulsion_sensor_vector, (dimW, numAgent, numAgent)).transpose(0, 2, 1)
    dU_repulsion_sensor = -np.sum(dU_repulsion_sensor_matrix, 2).T * APF_AGENT_REPULSE_GAIN

    num_agents = len(robots_positions)
    dim = 3
    dU_repulsion_sensor_obstacle = np.zeros((num_agents, dim))
    esdf_distances = np.zeros(num_agents)
    esdf_gradients = np.zeros((num_agents, dim))
    for i, pos in enumerate(robots_positions):
        d = esdf_map.get_esdf(pos)
        esdf_distances[i] = d - Rradius
        grad = esdf_map.compute_gradient(pos)
        esdf_gradients[i] = grad if grad is not None else np.zeros(dim)
    in_repulsion_zone = esdf_distances <= r_repulsion_obstacle
    affected_agents = np.where(in_repulsion_zone)[0]
    if affected_agents.size > 0:
        safe_dists = np.maximum(esdf_distances[affected_agents], APF_ESDF_SAFE_DIST_MIN)
        epsilon = 1e-10
        scale_factors = (1 / (safe_dists + epsilon) - 1 / r_repulsion_obstacle) / (safe_dists**3 + epsilon)
        repulsion_vectors = -scale_factors[:, np.newaxis] * esdf_gradients[affected_agents]
        np.add.at(dU_repulsion_sensor_obstacle, (affected_agents, slice(None)), repulsion_vectors)
    dU_repulsion_sensor_obstacle *= APF_ESDF_REPULSE_GAIN

    Dist_sensor_boundary_Left = robots_positions[:, 0] - xa + minDistance - Rradius
    Dist_sensor_boundary_Right = xb - robots_positions[:, 0] + minDistance - Rradius
    Dist_sensor_boundary_Top = yb - robots_positions[:, 1] + minDistance - Rradius
    Dist_sensor_boundary_Bottom = robots_positions[:, 1] - ya + minDistance - Rradius
    Dist_sensor_boundary_Front = robots_positions[:, 2] - za + minDistance - Rradius
    Dist_sensor_boundary_Back = zb - robots_positions[:, 2] + minDistance - Rradius
    Diff_sensor_boundary_Left = np.array([Dist_sensor_boundary_Left, np.zeros(numAgent), np.zeros(numAgent)]).T
    Diff_sensor_boundary_Right = np.array([-Dist_sensor_boundary_Right, np.zeros(numAgent), np.zeros(numAgent)]).T
    Diff_sensor_boundary_Top = np.array([np.zeros(numAgent), -Dist_sensor_boundary_Top, np.zeros(numAgent)]).T
    Diff_sensor_boundary_Bottom = np.array([np.zeros(numAgent), Dist_sensor_boundary_Bottom, np.zeros(numAgent)]).T
    Diff_sensor_boundary_Front = np.array([np.zeros(numAgent), np.zeros(numAgent), Dist_sensor_boundary_Front]).T
    Diff_sensor_boundary_Back = np.array([np.zeros(numAgent), np.zeros(numAgent), -Dist_sensor_boundary_Back]).T

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

    force_limit = 300000.0
    if not idx_Left.shape[0] == 0:
        dU_repulsion_Left[idx_Left, :] = -(1 / Dist_sensor_boundary_Left[idx_Left] - 1 / r_repulsion_obstacle)[
            :, np.newaxis
        ] * (Dist_sensor_boundary_Left[idx_Left] ** (-3))[:, np.newaxis] * Diff_sensor_boundary_Left[idx_Left, :]
        dU_repulsion_Left = np.clip(dU_repulsion_Left, -force_limit, force_limit)
    if not idx_Right.shape[0] == 0:
        dU_repulsion_Right[idx_Right, :] = -(
            1 / Dist_sensor_boundary_Right[idx_Right] - 1 / r_repulsion_obstacle
        )[:, np.newaxis] * (Dist_sensor_boundary_Right[idx_Right] ** (-3))[:, np.newaxis] * Diff_sensor_boundary_Right[
            idx_Right, :
        ]
        dU_repulsion_Right = np.clip(dU_repulsion_Right, -force_limit, force_limit)
    if not idx_Top.shape[0] == 0:
        dU_repulsion_Top[idx_Top, :] = -(1.0 / Dist_sensor_boundary_Top[idx_Top] - 1 / r_repulsion_obstacle)[
            :, np.newaxis
        ] * (Dist_sensor_boundary_Top[idx_Top] ** (-3))[:, np.newaxis] * Diff_sensor_boundary_Top[idx_Top, :]
        dU_repulsion_Top = np.clip(dU_repulsion_Top, -force_limit, force_limit)
    if not idx_Bottom.shape[0] == 0:
        dU_repulsion_Bottom[idx_Bottom, :] = -(
            1 / Dist_sensor_boundary_Bottom[idx_Bottom] - 1 / r_repulsion_obstacle
        )[:, np.newaxis] * (Dist_sensor_boundary_Bottom[idx_Bottom] ** (-3))[:, np.newaxis] * Diff_sensor_boundary_Bottom[
            idx_Bottom, :
        ]
        dU_repulsion_Bottom = np.clip(dU_repulsion_Bottom, -force_limit, force_limit)
    if not idx_Front.shape[0] == 0:
        dU_repulsion_Front[idx_Front, :] = -(
            1 / Dist_sensor_boundary_Front[idx_Front] - 1 / r_repulsion_obstacle
        )[:, np.newaxis] * (Dist_sensor_boundary_Front[idx_Front] ** (-3))[:, np.newaxis] * Diff_sensor_boundary_Front[
            idx_Front, :
        ]
        dU_repulsion_Front = np.clip(dU_repulsion_Front, -force_limit, force_limit)
    if not idx_Back.shape[0] == 0:
        dU_repulsion_Back[idx_Back, :] = -(1 / Dist_sensor_boundary_Back[idx_Back] - 1 / r_repulsion_obstacle)[
            :, np.newaxis
        ] * (Dist_sensor_boundary_Back[idx_Back] ** (-3))[:, np.newaxis] * Diff_sensor_boundary_Back[idx_Back, :]
        dU_repulsion_Back = np.clip(dU_repulsion_Back, -force_limit, force_limit)

    dU_repulsion_sensor_boundary = (
        dU_repulsion_Left
        + dU_repulsion_Right
        + dU_repulsion_Top
        + dU_repulsion_Bottom
        + dU_repulsion_Front
        + dU_repulsion_Back
    ) * APF_BOUNDARY_REPULSE_GAIN

    att = dU_sensor_gmm
    rep = dU_repulsion_sensor + dU_repulsion_sensor_obstacle + dU_repulsion_sensor_boundary
    att_n = np.linalg.norm(att, axis=1, keepdims=True)
    rep_n = np.linalg.norm(rep, axis=1, keepdims=True)
    att_u = np.divide(att, att_n, out=np.zeros_like(att), where=(att_n > 1e-10))
    rep_u = np.divide(rep, rep_n, out=np.zeros_like(rep), where=(rep_n > 1e-10))
    ba = float(APF_ATT_REP_UNIT_BLEND)
    dU = ba * att_u + (1.0 - ba) * rep_u
    comb_n = np.linalg.norm(dU, axis=1, keepdims=True)
    dU = np.divide(dU, np.maximum(comb_n, 1e-10))
    oppose = comb_n.ravel() < 1e-8
    if np.any(oppose):
        for j in np.flatnonzero(oppose):
            if att_n[j, 0] > 1e-10:
                dU[j, :] = att_u[j, :]
            elif rep_n[j, 0] > 1e-10:
                dU[j, :] = rep_u[j, :]
    U = U_sensor_gmm + gamma * U_sensor

    dists = np.asarray(esdf_map.get_esdf(robots_positions), dtype=float).reshape(-1)
    if np.any(dists <= 0):
        global _last_obstacle_warning_time
        now = time.time()
        if now - _last_obstacle_warning_time >= APF_OBSTACLE_WARN_PERIOD_SEC:
            _last_obstacle_warning_time = now
            print(
                "There are some sensors in the obstacle areas!!! "
                f"count={int(np.sum(dists <= 0))}, min_esdf={float(np.min(dists)):.4f}"
            )

    SensorPos_next = robots_positions - dU * max_Velocity
    dists = esdf_map.get_esdf(SensorPos_next)
    indexSensor = np.where(dists <= 0)[0]
    k = 1
    while k <= 3 and not indexSensor.shape[0] == 0:
        Velocity = max_Velocity * (1 / 2) ** k
        SensorPos_next[indexSensor] = robots_positions[indexSensor] - dU[indexSensor] * Velocity
        dists = esdf_map.get_esdf(SensorPos_next)
        indexSensor = np.where(dists <= 0)[0]
        k += 1

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

    SensorPos_next[:, 0] = np.minimum(
        np.maximum(SensorPos_next[:, 0], xa + minDistance + 0.5 * Rdiameter),
        xb - minDistance - 0.5 * Rdiameter,
    )
    SensorPos_next[:, 1] = np.minimum(
        np.maximum(SensorPos_next[:, 1], ya + minDistance + 0.5 * Rdiameter),
        yb - minDistance - 0.5 * Rdiameter,
    )
    SensorPos_next[:, 2] = np.minimum(
        np.maximum(SensorPos_next[:, 2], za + minDistance + 0.5 * Rdiameter),
        zb - minDistance - 0.5 * Rdiameter,
    )

    final_dists = np.asarray(esdf_map.get_esdf(SensorPos_next), dtype=float).reshape(-1)
    final_bad = np.where(final_dists <= 0)[0]
    if final_bad.size > 0:
        original_dists = np.asarray(esdf_map.get_esdf(robots_positions), dtype=float).reshape(-1)
        for ii in final_bad:
            if original_dists[ii] > 0:
                SensorPos_next[ii, :] = robots_positions[ii, :]
                continue
            grad = np.asarray(esdf_map.compute_gradient(SensorPos_next[ii, :]), dtype=float).reshape(-1)
            if grad.shape[0] < 3:
                continue
            gn = float(np.linalg.norm(grad[:3]))
            if gn < 1e-12:
                continue
            gu = grad[:3] / gn
            for step_scale in (1.0, 1.5, 2.0, 3.0):
                trial = SensorPos_next[ii, :] + gu * max_Velocity * step_scale
                trial[0] = min(max(trial[0], xa + minDistance + 0.5 * Rdiameter), xb - minDistance - 0.5 * Rdiameter)
                trial[1] = min(max(trial[1], ya + minDistance + 0.5 * Rdiameter), yb - minDistance - 0.5 * Rdiameter)
                trial[2] = min(max(trial[2], za + minDistance + 0.5 * Rdiameter), zb - minDistance - 0.5 * Rdiameter)
                if float(np.asarray(esdf_map.get_esdf(trial)).reshape(-1)[0]) > 0:
                    SensorPos_next[ii, :] = trial
                    break

    U_sensor_next_gmm = np.zeros(numAgent)
    for l in range(len(next_means)):
        sigma = Sigma[l] + sigma_k
        mu = Mu[l]
        weight = Weight[l]
        GM_gmm = multivariate_normal.pdf(SensorPos_next, mean=mu, cov=sigma)
        U_sensor_next_gmm += weight * GM_gmm
    U_sensor_next_gmm = -np.mean(U_sensor_next_gmm)

    Diff_sensor_next = np.zeros((numAgent**2, dimW))
    for n in range(dimW):
        SensorPos_next_vector = SensorPos_next[:, n]
        Diff_sensor_next_matrix = np.expand_dims(SensorPos_next_vector, axis=1) - np.expand_dims(
            SensorPos_next_vector, axis=1
        ).T
        Diff_sensor_next[:, n] = Diff_sensor_next_matrix.flatten(order="F")

    mu0 = np.zeros(dimW)
    sigma = 2 * sigma_k
    GM_sensor_next_vector = multivariate_normal.pdf(Diff_sensor_next, mean=mu0, cov=sigma)
    U_sensor_next = 0.5 * np.mean(GM_sensor_next_vector)

    U_next = U_sensor_next_gmm + gamma * U_sensor_next
    J_rate = U - U_next
    robots_positions_next = SensorPos_next
    return robots_positions_next, J_rate, U, U_next


def get_3d_trajectories(current_points, points_list):
    """三维轨迹分割函数（无投影）。"""

    def find_closest_index(query_point, trajectory):
        distances = cdist(trajectory, np.array([query_point]))
        return np.argmin(distances)

    trajectories = []
    num_robots = len(points_list)
    for i in range(num_robots):
        traj_3d = np.array(points_list[i]).reshape(-1, 3)
        closest_idx = find_closest_index(current_points[i], traj_3d)
        segmented_traj = traj_3d[closest_idx:]
        trajectories.append(segmented_traj)
    return trajectories
