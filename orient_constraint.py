import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse.linalg import expm
from mpl_toolkits.mplot3d import Axes3D


# Skew-symmetric matrix function
def skew(vector):
    return np.array(
        [
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0],
        ]
    )


# Define the CBF
def h(R, e_i, theta_i):
    return e_i.T @ R @ e_i - np.cos(theta_i)


# Define the objective function for control
def objective(u, u_nominal):
    return cp.norm(u - u_nominal) ** 2


# Define the CBF constraint
def cbf_constraint(u, R, e_i, theta, alpha):
    e_i_skew = skew(e_i)
    h_val = h(R, e_i, theta)
    return -e_i.T @ R @ e_i_skew @ u + alpha * h_val


# Simulation parameters
R = np.eye(3)  # Initial orientation matrix
R_init = R.copy()

e_i = np.array([1.0, 0, 0])  # Considered column of the orientation matrix
omega = np.array([3.0, 0.0, 0.0])  # Initial angular velocity
theta = np.array([0.6, 0, 0])  # Initial angle
theta_i = theta[0]  # Considered angle for the barrier function
k = 10.0  # Gain for the barrier function
dt = 0.01  # Time step
T = 1.0  # Simulation duration

# Store results for plotting
time = np.arange(0, T, dt)
x_trajectory = [0.0]
x_nocbf_trajectory = [0.0]
x_des = R.copy()
h_values = [h(R, e_i, theta_i)]
u_values = []

# Store the rotation matrix dynamics for 3D plotting
R_dynamics = []

# Simulation loop
for t in tqdm(time[1:]):
    u_nominal = omega
    u = cp.Variable(3)

    # Define the objective function (quadratic)
    obj = cp.Minimize(objective(u, u_nominal))

    # Define the CBF constraint
    constraints = [cbf_constraint(u, R, e_i, theta_i, k) >= 0]

    # Define the problem and solve it
    prob = cp.Problem(obj, constraints)
    prob.solve()

    if prob.status != cp.OPTIMAL:
        print(f"Optimization failed at time {t}: {prob.status}")
        break

    # Get the optimized control input
    u_opt = u.value

    # Update state using integration
    R_nocbf = R @ expm(dt * skew(omega))

    # Ensure the resulting matrix is still a valid rotation matrix (orthonormal)
    U, _, Vt = np.linalg.svd(R_nocbf)
    R_nocbf = U @ Vt

    R = R @ expm(dt * skew(u_opt))

    # Ensure the resulting matrix is still a valid rotation matrix (orthonormal)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    # Compute angle from the initial orientation matrix to the current orientation matrix
    init_vec = R_init[:, 0]
    cur_vec = R[:, 0]
    angle = np.arccos(
        np.dot(init_vec, cur_vec) / (np.linalg.norm(init_vec) * np.linalg.norm(cur_vec))
    )
    angle_nocbf = angle

    # Record the results
    x_trajectory.append(angle.copy())
    x_nocbf_trajectory.append(angle_nocbf.copy())
    h_values.append(h(R, e_i, theta_i))
    u_values.append(u_opt)

    # Store the rotation matrix for 3D visualization
    R_dynamics.append(R)

# Convert results to arrays for plotting
x_trajectory = np.array(x_trajectory)
x_nocbf_trajectory = np.array(x_nocbf_trajectory)
h_values = np.array(h_values)
R_dynamics = np.array(R_dynamics)

# Plot the results
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# 1. Plot the state trajectory with and without CBF over time
axs[0, 0].plot(time, x_trajectory, "-o", label="State trajectory with CBF", color="b")
axs[0, 0].plot(
    time,
    x_nocbf_trajectory,
    "--",
    label="State trajectory without CBF",
    color="orange",
)
axs[0, 0].axhline(theta_i, color="r", linestyle="--", label="Boundary (h(x) = 0)")
axs[0, 0].set_xlabel("Time (s)", fontsize=20)
axs[0, 0].set_ylabel("Roll (x[0])", fontsize=20)
axs[0, 0].set_title("State Trajectory with and without CBF (gamma = {k})", fontsize=20)
axs[0, 0].legend(fontsize=20)
axs[0, 0].axis("equal")

# 2. Plot the barrier function h(x)
axs[0, 1].plot(time, h_values, label="h(x) with CBF", color="b")
axs[0, 1].axhline(0, color="r", linestyle="--", label="Boundary (h(x) = 0)")
axs[0, 1].set_xlabel("Time (s)", fontsize=20)
axs[0, 1].set_ylabel("Barrier Function h(x)", fontsize=20)
axs[0, 1].set_title(
    "Evolution of the Barrier Function h(x) over Time with CBF Constraint", fontsize=20
)
axs[0, 1].legend(fontsize=20)

# 3. Plot the control input values
u_values = np.array(u_values)
axs[1, 0].plot(time[1:], u_values[:, 0], label="u[0] (x-component)", color="lime")
axs[1, 0].plot(time[1:], u_values[:, 1], label="u[1] (y-component)", color="violet")
axs[1, 0].set_xlabel("Time (s)", fontsize=20)
axs[1, 0].set_ylabel("Control Input Magnitude", fontsize=20)
axs[1, 0].set_title(
    "Control Inputs over Time: x- and y-Components with CBF Constraint", fontsize=20
)
axs[1, 0].legend(fontsize=20)

# 4. Plot 3D visualization of the rotation matrix dynamics
ax = fig.add_subplot(2, 2, 4, projection="3d")
time3D = time[:-1]  # Time array excluding the first time step
R_dynamics = np.array(R_dynamics)
print(R_dynamics.shape)

ax.plot(
    time3D,
    R_dynamics[:, 0, 0],
    R_dynamics[:, 0, 1],
    label="R[0,0] vs R[0,1]",
    color="b",
)
ax.plot(
    time3D,
    R_dynamics[:, 1, 0],
    R_dynamics[:, 1, 1],
    label="R[1,0] vs R[1,1]",
    color="orange",
)
ax.plot(
    time3D,
    R_dynamics[:, 2, 0],
    R_dynamics[:, 2, 1],
    label="R[2,0] vs R[2,1]",
    color="green",
)

ax.set_xlabel("Time (s)", fontsize=20)
ax.set_ylabel("Matrix Component", fontsize=20)
ax.set_zlabel("Rotation Matrix Values", fontsize=20)
ax.set_title("3D Visualization of Rotation Matrix Dynamics", fontsize=20)
ax.legend(fontsize=15)

# Show the plots with tight layout
plt.tight_layout()
plt.show()
