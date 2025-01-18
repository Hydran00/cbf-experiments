import numpy as np
import cvxpy as cp
from tqdm import tqdm
from scipy.sparse.linalg import expm
from scipy.spatial.transform import Rotation as R
from plot_cone_constraints import plot_cone_constraints


# Skew-symmetric matrix function
def skew(vector):
    return np.array(
        [
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0],
        ]
    )


def angle_between_vectors(v1, v2):
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# Define the objective function for control
def objective(u, u_nominal):
    return cp.norm(u - u_nominal) ** 2


# Define the CBF
def h(R, e_w, e_i, theta_i):
    return e_w.T @ R.T @ e_i - np.cos(theta_i)


# Define the CBF constraint
def cbf_constraint(u, R, e_w, e_i, theta, alpha):
    # return -e_i.T @ R @ skew(e_i) @ u + alpha * h(R, e_i, theta)
    e_w_skew = skew(e_w)
    h_val = h(R, e_w, e_i, theta)
    return -e_i.T @ R @ e_w_skew @ u + alpha * h_val


# x_test = np.array(
#     [
#         [0.9396926, -0.0000000, 0.3420202],
#         [0.1169778, 0.9396926, -0.3213938],
#         [-0.3213938, 0.3420202, 0.8830222],
#     ]
# )
# e_test = np.array(x_test[:, 0])
# print(e_test.shape)
# e_i_test_skew = skew(e_test)
# print(e_i_test_skew.shape)

# theta_test = 0.4
# gamma_test = 10.0
# print(h(x_test, e_test, theta_test))
# print(-e_test.T @ x_test @ e_i_test_skew)


# Simulation parameters
omega = np.array([-0.3, 0.36, -0.5])  # Angular velocity
x = R.from_euler("xyz", [-0.2, -0.3, 0.8]).as_matrix()
x = np.eye(3)
x_nocbf = x.copy()
e_w = np.eye(3)  # world frame
e_ref = x  # reference frame to which we define the angle limits
theta = np.array([0.3, 0.3, 0.6])  # angle limits

k = 1.0  # Gain for the barrier function
dt = 0.001  # Time step
T = 12.2  # Simulation duration

# Store results for plotting
time = np.arange(0, T, dt)
# pre allocate the matrices
log_x_dynamics = np.zeros((3, 3, len(time)))
log_x_trajectory = np.zeros((3, 3, len(time)))
log_x_nocbf_trajectory = np.zeros((3, len(time)))
log_h = np.zeros((3, len(time)))
log_u = np.zeros((3, len(time)))
x[1, 1] = -1.0
x[2, 2] = -1.0
x = x @ expm(dt * skew([0.0, 0.5, 0.0]))
print("int :\n", x)

print(e_w[:, 0].T @ x.T @ e_ref[:, 0])
constraints = ["", "", ""]
# Simulation loop
for step, t in enumerate(tqdm(time)):

    u_nominal = omega
    u = cp.Variable(3)

    for i in range(3):
        # log_x_trajectory[i, step] = angle_between_vectors(e[:, i], x @ e[:, i])
        # log_x_nocbf_trajectory[i, step] = angle_between_vectors(e[:, i], x @ e[:, i])
        log_h[i, step] = h(x, e_w[:, i], e_ref[:, i], theta[i])
        # Define the CBF constraint
        constraints[i] = cbf_constraint(u, x, e_w[:, i], e_ref[:, i], theta[i], k) >= 0
    log_x_trajectory[:, :, step] = x.copy()

    # Define the objective function (quadratic)
    obj = cp.Minimize(objective(u, u_nominal))

    # Define the problem and solve it
    prob = cp.Problem(obj, constraints)
    prob.solve()

    if prob.status != cp.OPTIMAL:
        print(f"Optimization failed at time {t}: {prob.status}")
        break

    # Get the optimized control input
    u_opt = u.value

    # integrate the dynamics
    x = x @ expm(dt * skew(u_opt))
    x_nocbf = x_nocbf @ expm(dt * skew(omega))

    # Ensure the resulting matrix is still a valid rotation matrix (orthonormal)
    U, _, Vt = np.linalg.svd(x_nocbf)
    x_nocbf = U @ Vt
    U, _, Vt = np.linalg.svd(x)
    x = U @ Vt

    # Compute angle from the initial orientation matrix to the current orientation matrix
    log_u[:, step] = u_opt

    # Store the rotation matrix for 3D visualization
    log_x_dynamics[:, :, step] = x
print("Final angle between vectors:")
print("x: ", angle_between_vectors(e_ref[0], x[:, 0]))
print("y: ", angle_between_vectors(e_ref[1], x[:, 1]))
print("z: ", angle_between_vectors(e_ref[2], x[:, 2]))
plot_cone_constraints(time, log_x_trajectory, log_h, log_u, e_ref, theta)
