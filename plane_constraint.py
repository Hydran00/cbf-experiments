import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm


# Define the CBF
def h(x, n, p):
    return np.dot(n, x - p)


# Define the objective; track the nominal control input
def objective(u, u_nominal):
    return (
        cp.norm(u - u_nominal) ** 2
    )  # Minimize the squared difference from the nominal control input


# Define the CBF constraint
def cbf_constraint(u, x, n, p, k):
    return cp.matmul(n, u) + k * h(x, n, p)


# Simulation parameters
n = np.array([1.0, 0.0])  # Normal vector of the plane
p = np.array([2.0, 1.0])  # A point on the plane
k = 1.0  # Gain for the barrier function
x = np.array([2.5, 0.5])  # Initial state
dt = 0.1  # Time step
T = 100.0  # Simulation duration
w = 1.0  # Frequency of the nominal control input

# Store results for plotting
time = np.arange(0, T, dt)

x_trajectory = [x.copy()]
x_nocbf = x.copy()
x_nocbf_trajectory = [x.copy()]
x_des = x.copy()

h_values = [h(x, n, p)]
u_values = []

# Simulation loop
for t in tqdm(time[1:]):
    # Nominal control input (sinusoidal trajectory)
    x_des = np.array([x_des[0] + np.cos(w * t), x_des[1] + 0.1])
    u_nominal = (x_des - x) / dt

    # Create the CVXPY variable for control input
    u = cp.Variable(2)  # Control input as a 2D variable

    # Define the objective function (quadratic)
    obj = cp.Minimize(objective(u, u_nominal))

    # Define the CBF constraint
    constraints = [cbf_constraint(u, x, n, p, k) >= 0]

    # Define the problem and solve it
    prob = cp.Problem(obj, constraints)
    prob.solve()

    if prob.status != cp.OPTIMAL:
        print(f"Optimization failed at time {t}: {prob.status}")
        break

    # Get the optimized control input
    u_opt = u.value

    # Update state using Euler integration
    x_nocbf = x + u_nominal * dt
    x = x + u_opt * dt

    # Record the results
    x_trajectory.append(x.copy())
    x_nocbf_trajectory.append(x_nocbf.copy())
    h_values.append(h(x, n, p))
    u_values.append(u_opt)

# Convert results to arrays for plotting
x_trajectory = np.array(x_trajectory)
x_nocbf_trajectory = np.array(x_nocbf_trajectory)
h_values = np.array(h_values)
print(x_trajectory.size)
print(x_nocbf_trajectory.size)

# Plot the results
plt.figure(figsize=(12, 8))

# 1. Plot the state trajectory with and without CBF
plt.subplot(2, 2, 1)
plt.plot(
    x_trajectory[:, 1], x_trajectory[:, 0], "-o", label="State trajectory with CBF"
)
plt.plot(
    x_nocbf_trajectory[:, 1],
    x_nocbf_trajectory[:, 0],
    "--",
    label="State trajectory without CBF",
)
plt.axhline(p[0], color="r", linestyle="--", label="Plane Boundary")
plt.xlabel("Y (x[1])", fontsize=20)
plt.ylabel("X (x[0])", fontsize=20)
plt.title(
    f"State Trajectory with and without CBF (gamma = {k}): Comparison",
    fontsize=20,
)
plt.legend(fontsize=20)
plt.axis("equal")  # set scale to be equal

# 2. Plot the barrier function h(x)
plt.subplot(2, 2, 2)
plt.plot(time, h_values, label="h(x) with CBF", color="b")
plt.axhline(0, color="r", linestyle="--", label="Boundary (h(x) = 0)")
plt.xlabel("Time (s)", fontsize=20)
plt.ylabel("Barrier Function h(x)", fontsize=20)
plt.title(
    "Evolution of the Barrier Function h(x) over Time with CBF Constraint",
    fontsize=20,
)
plt.legend(fontsize=20)

# 3. Plot the control input values (x and y components)
u_values = np.array(u_values)
plt.subplot(2, 2, 3)
plt.plot(time[1:], u_values[:, 0], label="u[0] (x-component)", color="lime")
plt.plot(time[1:], u_values[:, 1], label="u[1] (y-component)", color="violet")
plt.xlabel("Time (s)", fontsize=20)
plt.ylabel("Control Input Magnitude", fontsize=20)
plt.title(
    "Control Inputs over Time: x- and y-Components with CBF Constraint",
    fontsize=20,
)
plt.legend(fontsize=20)

# Show the plots with tight layout
plt.tight_layout()
plt.show()
