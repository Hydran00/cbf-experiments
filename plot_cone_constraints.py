from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def plot_cone_constraints(time, x, x_init, h, u, theta):
    # Create a figure with multiple subplots (2 rows, 2 columns)
    fig = plt.figure(figsize=(16, 8))

    # Plot the barrier function values over time (subplot 1)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(
        time, h[0, :], label=f"Barrier function (axis X)", linewidth=2.5, color="red"
    )
    ax1.plot(
        time, h[1, :], label=f"Barrier function (axis Y)", linewidth=2.5, color="green"
    )
    ax1.plot(
        time, h[2, :], label=f"Barrier function (axis Z)", linewidth=2.5, color="blue"
    )

    ax1.axhline(0, color="black", linestyle="--", label="h = 0", linewidth=2.5)
    ax1.set_xlabel("Time (s)", fontsize=20)
    ax1.set_ylabel("h(R)", fontsize=20)
    ax1.set_title("CBF Values Over Time", fontsize=24)
    ax1.legend(fontsize=16)
    ax1.grid(True)

    # Plot the control inputs over time (subplot 2)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(
        time, u[0, :], label=f"Control input (omega X)", linewidth=2.5, color="red"
    )
    ax2.plot(
        time, u[1, :], label=f"Control input (omega Y)", linewidth=2.5, color="green"
    )
    ax2.plot(
        time, u[2, :], label=f"Control input (omega X)", linewidth=2.5, color="blue"
    )

    ax2.set_xlabel("Time (s)", fontsize=20)
    ax2.set_ylabel("Control input (rad/s)", fontsize=20)
    ax2.set_title("Control Inputs Over Time", fontsize=24)
    ax2.legend(fontsize=16)
    ax2.grid(True)

    # Function to plot a half-cone along a specified axis
    def plot_half_cone(ax, theta, direction, color, alpha=0.3):
        cone_height = 1.0  # Height of the cone (unit length for visualization)
        u, v = np.mgrid[0 : 2 * np.pi : 100j, 0:cone_height:50j]

        # Create half-cone surface (opening angle based on theta_limit)
        radius = v * np.tan(theta)
        x = radius * np.cos(u)
        y = radius * np.sin(u)
        z = v

        # Rotate and translate the half-cone to point along the given direction
        if direction == "x":
            ax.plot_surface(
                z, x, y, color=color, alpha=alpha
            )  # Positive x-direction only
        elif direction == "y":
            ax.plot_surface(
                x, z, y, color=color, alpha=alpha
            )  # Positive y-direction only
        elif direction == "z":
            ax.plot_surface(
                x, y, z, color=color, alpha=alpha
            )  # Positive z-direction only

    # Plot the 3D dynamics (subplot 3)
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")

    # Plot initial frame
    ax3.quiver(
        0,
        0,
        0,
        x_init[0, 0],
        x_init[1, 0],
        x_init[2, 0],
        color="r",
        linewidth=2.5,
        label="Initial frame (x-axis)",
    )
    ax3.quiver(
        0,
        0,
        0,
        x_init[0, 1],
        x_init[1, 1],
        x_init[2, 1],
        color="g",
        linewidth=2.5,
        label="Initial frame (y-axis)",
    )
    ax3.quiver(
        0,
        0,
        0,
        x_init[0, 2],
        x_init[1, 2],
        x_init[2, 2],
        color="b",
        linewidth=2.5,
        label="Initial frame (z-axis)",
    )

    # Plot final frame
    ax3.quiver(
        0,
        0,
        0,
        x[0, 0, -1],
        x[1, 0, -1],
        x[2, 0, -1],
        color="r",
        linestyle="dashed",
        linewidth=2.5,
        label="Final frame (x-axis)",
    )
    ax3.quiver(
        0,
        0,
        0,
        x[0, 1, -1],
        x[1, 1, -1],
        x[2, 1, -1],
        color="g",
        linestyle="dashed",
        linewidth=2.5,
        label="Final frame (y-axis)",
    )
    ax3.quiver(
        0,
        0,
        0,
        x[0, 2, -1],
        x[1, 2, -1],
        x[2, 2, -1],
        color="b",
        linestyle="dashed",
        linewidth=2.5,
        label="Final frame (z-axis)",
    )

    # Plot the trajectory of x on the unit sphere (projection of x)
    for t in range(0, x.shape[2], 10):
        norm_x = np.linalg.norm(x[:, 0, t])  # Normalize each x to have a radius = 1
        x_proj = x[:, 0, t] / norm_x  # Projection on unit sphere

        norm_y = np.linalg.norm(x[:, 1, t])
        y_proj = x[:, 1, t] / norm_y

        norm_z = np.linalg.norm(x[:, 2, t])
        z_proj = x[:, 2, t] / norm_z

        # plot projection points
        ax3.scatter(x_proj[0], x_proj[1], x_proj[2], color="r", s=30)
        ax3.scatter(y_proj[0], y_proj[1], y_proj[2], color="g", s=30)
        ax3.scatter(z_proj[0], z_proj[1], z_proj[2], color="b", s=30)

    # Add half-cones for each axis
    plot_half_cone(ax3, theta[0], "x", "red", alpha=0.3)
    plot_half_cone(ax3, theta[1], "y", "green", alpha=0.3)
    plot_half_cone(ax3, theta[2], "z", "blue", alpha=0.3)

    # Set axis limits and labels
    ax3.set_xlim([-1, 1])
    ax3.set_ylim([-1, 1])
    ax3.set_zlim([-1, 1])
    ax3.set_xlabel("X", fontsize=20)
    ax3.set_ylabel("Y", fontsize=20)
    ax3.set_zlabel("Z", fontsize=20)
    ax3.set_title("3D Visualization of Rotation Dynamics with Cone Limits", fontsize=24)
    ax3.legend(fontsize=16)

    # Adjust layout and show the figure with all plots
    plt.tight_layout()
    plt.show()
