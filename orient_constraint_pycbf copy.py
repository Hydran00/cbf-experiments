import jax.numpy as jnp
from cbfpy import CBF, CBFConfig
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# Create a config class for your problem inheriting from the CBFConfig class
class PlaneConstraintCBF(CBFConfig):
    def __init__(self, n, p):
        super().__init__(
            # Define the state and control dimensions
            n=2,
            m=2,
            # Define control limits (if desired)
            u_min=None,
            u_max=None,
        )
        self.n = n
        self.p = p

    def alpha(self, h):
        return 10.0 * h

    # Define the control-affine dynamics functions `f` and `g` for your system
    def f(self, z):
        return jnp.array([0.0, 0.0])

    def g(self, z):
        return jnp.eye(2)

    # Define the barrier function `h`
    # The *relative degree* of this system is 2, so, we'll use the h_2 method
    def h_1(self, z):
        return jnp.array([jnp.dot(jnp.array([1.0, 0.0]), z - jnp.array([2.0, 1.0]))])


def main():
    # Simulation parameters
    n = jnp.array([1.0, 0.0])  # Normal vector of the plane
    p = jnp.array([2.0, 1.0])  # A point on the plane
    # Create the CBF object
    config = PlaneConstraintCBF(n, p)
    plane_cbf = CBF.from_config(config)
    print("Starting CBF")

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

    h_values = [plane_cbf.h_1(x)]
    u_values = []
    # Simulation loop
    for t in tqdm(time[1:]):
        # Nominal control input (sinusoidal trajectory)
        x_des = np.array([x_des[0] + np.cos(w * t), x_des[1] + 0.1])
        u_nominal = (x_des - x) / dt

        # retrieve control input
        u_opt = plane_cbf.safety_filter(x, u_nominal)

        # Update state using Euler integration
        x_nocbf = x + u_nominal * dt
        x = x + u_opt * dt

        # Record the results
        x_trajectory.append(x.copy())
        x_nocbf_trajectory.append(x_nocbf.copy())
        h_values.append(plane_cbf.h_1(x))
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


if __name__ == "__main__":
    main()
