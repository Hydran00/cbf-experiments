import jax.numpy as jnp
from cbfpy import CBF, CBFConfig
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# Create a config class for your problem inheriting from the CBFConfig class
class OrientationConstraintCBF(CBFConfig):
    def __init__(self):
        self.e = jnp.eye(3)  # Standard basis vectors
        self.theta = jnp.array([0.6, 0.6, 0.6])  # Threshold angles
        self.considered_column = 0  # Choose the basis vector for the constraint
        self.omega = jnp.zeros(3)  # Control input
        super().__init__(
            # Define the state and control dimensions
            n=9,  # State dimension (flattened 3x3 rotation matrix)
            m=3,  # Control dimension (angular velocity vector)
            # Define control limits (if desired)
            u_min=None,
            u_max=None,
        )

    def skew(self, vector):
        """
        Returns the skew-symmetric matrix of a vector.
        :param vector: A 3-element array representing a vector.
        :return: A 3x3 skew-symmetric matrix.
        """
        return jnp.array(
            [
                [0, -vector[2], vector[1]],
                [vector[2], 0, -vector[0]],
                [-vector[1], vector[0], 0],
            ]
        )

    def set_u(self, u):
        """
        Set the control input for the system.
        :param u: The control input for the system.
        """
        self.omega = u

    def alpha(self, h):
        """Class-K function for the CBF condition."""
        return 10.0 * h

    def f(self, z):
        """Drift dynamics of the system (no drift in this case)."""
        return jnp.zeros(9)

    # def g(self, z):
    #     """Control-affine dynamics matrix g(x)."""
    #     # Reconstruct the rotation matrix R from the state
    #     R = jnp.array([[z[0], z[1], z[2]], [z[3], z[4], z[5]], [z[6], z[7], z[8]]])
    #     # g(x) = R * [omega]_{\times}
    #     omega_skew = self.skew(self.omega)
    #     return R @ omega_skew  # The effect of control input on the state
    def g(self, z):
        """Control-affine dynamics matrix g(x)."""
        # Reconstruct the rotation matrix R from the state (which is flattened)
        R = jnp.array([[z[0], z[1], z[2]], [z[3], z[4], z[5]], [z[6], z[7], z[8]]])

        # Compute the skew-symmetric matrix of omega (angular velocity)
        omega_skew = self.skew(self.omega)

        # Calculate the control-affine term g(x) = R * omega_skew
        g_matrix = R @ omega_skew

        # Flatten the result to match the expected (9, 3) shape
        return g_matrix.flatten()

    def h_1(self, z):
        """Barrier function h(R)."""
        R = jnp.array([[z[0], z[1], z[2]], [z[3], z[4], z[5]], [z[6], z[7], z[8]]])
        e_i = self.e[:, self.considered_column]
        # h(R) = e_i^T R e_i - cos(theta)
        return e_i.T @ R @ e_i - jnp.cos(self.theta[self.considered_column])

    def h_dot(self, z):
        """Time derivative of the barrier function h(R)."""
        R = jnp.array([[z[0], z[1], z[2]], [z[3], z[4], z[5]], [z[6], z[7], z[8]]])
        e_i = self.e[:, self.considered_column]
        e_i_skew = self.skew(e_i)
        # h_dot(x) = -e_i^T R [e_i]_{\times} omega + alpha h(R)
        h_val = self.h_1(z)
        return -e_i.T @ R @ e_i_skew @ self.omega + self.alpha(h_val) * h_val


def main():
    # Create the CBF object
    config = OrientationConstraintCBF()
    orient_cbf = CBF.from_config(config)
    print("Starting CBF simulation")

    k = 1.0  # Gain for the barrier function
    x = np.eye(3)  # Initial state (identity rotation matrix)
    dt = 0.1  # Time step
    T = 10.0  # Simulation duration

    # Store results for plotting
    time = np.arange(0, T, dt)
    x_trajectory = [x.copy()]
    h_values = [orient_cbf.h_1(x)]
    u_values = []

    # Simulation loop
    for t in tqdm(time[1:]):
        # Nominal control input (zero for simplicity)
        u_nominal = np.zeros(3)
        config.set_u(u_nominal)
        # Retrieve control input using the safety filter
        u_opt = orient_cbf.safety_filter(x, u_nominal)

        # Update state using Euler integration for rotation matrix dynamics
        omega = u_opt  # Control input directly as angular velocity
        R = np.array([[x[0], x[1], x[2]], [x[3], x[4], x[5]], [x[6], x[7], x[8]]])
        R_dot = R @ config.skew(omega)  # Rotation matrix dynamics: R_dot = R [omega]_x
        x = x + (R_dot * dt  # Update state (flattened rotation matrix)

        # Record the results
        x_trajectory.append(x.copy())
        h_values.append(orient_cbf.h_1(x))
        u_values.append(u_opt)

    # Convert results to arrays for plotting
    x_trajectory = np.array(x_trajectory)
    h_values = np.array(h_values)
    u_values = np.array(u_values)

    # Plot the results
    plt.figure(figsize=(12, 8))

    # 1. Plot the barrier function h(x)
    plt.subplot(2, 1, 1)
    plt.plot(time, h_values, label="h(x) with CBF", color="b")
    plt.axhline(0, color="r", linestyle="--", label="Boundary (h(x) = 0)")
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Barrier Function h(x)", fontsize=14)
    plt.title(
        "Evolution of the Barrier Function h(x) over Time with CBF Constraint",
        fontsize=16,
    )
    plt.legend(fontsize=12)

    # 2. Plot the control input values
    plt.subplot(2, 1, 2)
    plt.plot(time[1:], u_values[:, 0], label="u[0] (x-component)", color="lime")
    plt.plot(time[1:], u_values[:, 1], label="u[1] (y-component)", color="violet")
    plt.plot(time[1:], u_values[:, 2], label="u[2] (z-component)", color="orange")
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Control Input Magnitude", fontsize=14)
    plt.title(
        "Control Inputs over Time: x, y, and z Components with CBF Constraint",
        fontsize=16,
    )
    plt.legend(fontsize=12)

    # Show the plots with tight layout
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
