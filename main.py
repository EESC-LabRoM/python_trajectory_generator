"""
Main script for generating quadrotor trajectories using differential flatness.
Converted from MATLAB implementation.
"""

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from traj_generator_casadi import traj_generator_casadi


def main():
    """Main function to generate and save quadrotor trajectory."""
    
    # File settings
    file_path = ''
    filename = 'figure_eight_v1_a05_yaw025.csv'
    
    # Simulation parameters
    dt = 0.1          # resolution
    tend = 40.0       # total time (should be greater than 10)
    
    # Options
    opt = {
        'zero_lateral_overload': True,   # true: no lateral overload, similar to quadrotors
                                        # false: zero pitch and roll
        'reset_terminal_att': False      # regulate terminal quaternion or not
    }
    
    # Plotting option
    make_plot = True
    
    # Define distorted time variable
    t = ca.SX.sym('t')
    
    # Time distortion for a smooth start
    k = 1
    t0 = 5
    t1 = tend - 5.0
    ts = ca.log(1 + ca.exp(k * (t - t0))) - ca.log(1 + ca.exp(k * (t - t1)))
    zvars = 1e-15 * ca.sin(ts)  # zero variable, somehow '0' does not work
    
    # Define algebraic equations of trajectories
    px = 2.5 * ca.cos(0.25 * ts) + 0.2
    py = 2.0 * ca.sin(0.5 * ts)
    pz = zvars + 1
    psi = 0.25 * ts  # heading
    
    # Generate trajectory
    print("Generating trajectory...")
    data = traj_generator_casadi(t, px, py, pz, psi, dt, tend, opt)
    
    # Save to CSV
    full_filename = os.path.join(file_path, filename)
    data.to_csv(full_filename, index=False)
    print(f"Trajectory saved to: {full_filename}")
    
    # Plotting
    if make_plot:
        plot_trajectory(data)
    
    return data


def plot_trajectory(data):
    """Plot the generated trajectory."""
    
    time = data['t'].values
    
    # Calculate magnitudes
    acc = np.sqrt(data['a_lin_x']**2 + data['a_lin_y']**2 + data['a_lin_z']**2)
    vel = np.sqrt(data['v_x']**2 + data['v_y']**2 + data['v_z']**2)
    rate = np.sqrt(data['w_x']**2 + data['w_y']**2 + data['w_z']**2)
    
    # Time series plots
    plt.figure(figsize=(12, 8))
    plt.suptitle('Reference Trajectory')
    
    plt.subplot(3, 1, 1)
    plt.plot(time, vel)
    plt.ylabel('Velocity [m/s]')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(time, acc)
    plt.ylabel('Acceleration [m/s²]')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(time, rate)
    plt.ylabel('Angular Rate [rad/s]')
    plt.xlabel('Time [s]')
    plt.grid(True)
    
    plt.tight_layout()
    
    # 3D trajectory plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(data['p_x'], data['p_y'], data['p_z'])
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('3D Reference Trajectory')
    ax.grid(True)
    
    # Make axes equal
    max_range = np.array([
        data['p_x'].max() - data['p_x'].min(),
        data['p_y'].max() - data['p_y'].min(),
        data['p_z'].max() - data['p_z'].min()
    ]).max() / 2.0
    
    mid_x = (data['p_x'].max() + data['p_x'].min()) * 0.5
    mid_y = (data['p_y'].max() + data['p_y'].min()) * 0.5
    mid_z = (data['p_z'].max() + data['p_z'].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()


if __name__ == "__main__":
    data = main()