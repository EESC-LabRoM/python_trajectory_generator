"""
Examples of different trajectory types using the Python trajectory generator.
"""

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from traj_generator_casadi import traj_generator_casadi


def generate_circular_trajectory():
    """Generate a simple circular trajectory."""
    
    print("Generating circular trajectory...")
    
    dt = 0.1
    tend = 20.0
    
    opt = {
        'zero_lateral_overload': True,
        'reset_terminal_att': False
    }
    
    t = ca.SX.sym('t')
    
    # Circular trajectory
    radius = 3.0
    omega = 0.5  # angular frequency
    
    px = radius * ca.cos(omega * t)
    py = radius * ca.sin(omega * t)
    pz = 2.0 + 0.0 * t  # constant height
    psi = 0.0 * t  # constant heading
    
    data = traj_generator_casadi(t, px, py, pz, psi, dt, tend, opt)
    data.to_csv('circular_trajectory.csv', index=False)
    print("Circular trajectory saved to: circular_trajectory.csv")
    
    return data


def generate_helix_trajectory():
    """Generate a helical (spiral) trajectory."""
    
    print("\nGenerating helix trajectory...")
    
    dt = 0.1
    tend = 30.0
    
    opt = {
        'zero_lateral_overload': True,
        'reset_terminal_att': False
    }
    
    t = ca.SX.sym('t')
    
    # Helix trajectory
    radius = 2.5
    omega = 0.3  # angular frequency
    climb_rate = 0.05  # vertical speed
    
    px = radius * ca.cos(omega * t)
    py = radius * ca.sin(omega * t)
    pz = 1.0 + climb_rate * t  # climbing height
    psi = 0.1 * t  # slowly rotating heading
    
    data = traj_generator_casadi(t, px, py, pz, psi, dt, tend, opt)
    data.to_csv('helix_trajectory.csv', index=False)
    print("Helix trajectory saved to: helix_trajectory.csv")
    
    return data


def generate_lemniscate_trajectory():
    """Generate a lemniscate (infinity symbol) trajectory."""
    
    print("\nGenerating lemniscate trajectory...")
    
    dt = 0.1
    tend = 25.0
    
    opt = {
        'zero_lateral_overload': True,
        'reset_terminal_att': False
    }
    
    t = ca.SX.sym('t')
    
    # Lemniscate (figure-8) trajectory
    a = 3.0  # scale factor
    omega = 0.4
    
    # Parametric equations for lemniscate
    cos_t = ca.cos(omega * t)
    sin_t = ca.sin(omega * t)
    
    px = a * cos_t / (1 + sin_t**2)
    py = a * cos_t * sin_t / (1 + sin_t**2)
    pz = 1.5 + 0.2 * ca.sin(0.1 * t)  # slight vertical oscillation
    psi = 0.05 * t  # slow heading change
    
    data = traj_generator_casadi(t, px, py, pz, psi, dt, tend, opt)
    data.to_csv('lemniscate_trajectory.csv', index=False)
    print("Lemniscate trajectory saved to: lemniscate_trajectory.csv")
    
    return data


def generate_square_trajectory():
    """Generate a square trajectory using smooth transitions."""
    
    print("\nGenerating square trajectory...")
    
    dt = 0.1
    tend = 20.0
    
    opt = {
        'zero_lateral_overload': True,
        'reset_terminal_att': False
    }
    
    t = ca.SX.sym('t')
    
    # Square trajectory using Fourier series approximation
    # This creates a smooth approximation to a square path
    size = 2.0
    n_harmonics = 5
    
    px = 0
    py = 0
    
    for n in range(1, n_harmonics + 1):
        if n % 2 == 1:  # odd harmonics only
            px += (4 * size / (n * np.pi)) * ca.cos(n * 0.5 * t)
            py += (4 * size / (n * np.pi)) * ca.sin(n * 0.5 * t)
    
    pz = 1.0 + 0.0 * t  # constant height
    psi = 0.0 * t  # constant heading
    
    data = traj_generator_casadi(t, px, py, pz, psi, dt, tend, opt)
    data.to_csv('square_trajectory.csv', index=False)
    print("Square trajectory saved to: square_trajectory.csv")
    
    return data


def plot_trajectories_comparison():
    """Create a comparison plot of different trajectory types."""
    
    # Generate all trajectories
    circular_data = generate_circular_trajectory()
    helix_data = generate_helix_trajectory()
    lemniscate_data = generate_lemniscate_trajectory()
    square_data = generate_square_trajectory()
    
    # Create comparison plots
    fig = plt.figure(figsize=(15, 10))
    
    # 3D plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(circular_data['p_x'], circular_data['p_y'], circular_data['p_z'], 'b-', label='Circular')
    ax1.plot(helix_data['p_x'], helix_data['p_y'], helix_data['p_z'], 'r-', label='Helix')
    ax1.plot(lemniscate_data['p_x'], lemniscate_data['p_y'], lemniscate_data['p_z'], 'g-', label='Lemniscate')
    ax1.plot(square_data['p_x'], square_data['p_y'], square_data['p_z'], 'm-', label='Square')
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_zlabel('Z [m]')
    ax1.legend()
    ax1.set_title('3D Trajectories')
    
    # XY plot
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(circular_data['p_x'], circular_data['p_y'], 'b-', label='Circular')
    ax2.plot(helix_data['p_x'], helix_data['p_y'], 'r-', label='Helix')
    ax2.plot(lemniscate_data['p_x'], lemniscate_data['p_y'], 'g-', label='Lemniscate')
    ax2.plot(square_data['p_x'], square_data['p_y'], 'm-', label='Square')
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.legend()
    ax2.set_title('XY Plane View')
    ax2.grid(True)
    ax2.axis('equal')
    
    # Velocity comparison
    ax3 = fig.add_subplot(2, 2, 3)
    circular_vel = np.sqrt(circular_data['v_x']**2 + circular_data['v_y']**2 + circular_data['v_z']**2)
    helix_vel = np.sqrt(helix_data['v_x']**2 + helix_data['v_y']**2 + helix_data['v_z']**2)
    lemniscate_vel = np.sqrt(lemniscate_data['v_x']**2 + lemniscate_data['v_y']**2 + lemniscate_data['v_z']**2)
    square_vel = np.sqrt(square_data['v_x']**2 + square_data['v_y']**2 + square_data['v_z']**2)
    
    ax3.plot(circular_data['t'], circular_vel, 'b-', label='Circular')
    ax3.plot(helix_data['t'], helix_vel, 'r-', label='Helix')
    ax3.plot(lemniscate_data['t'], lemniscate_vel, 'g-', label='Lemniscate')
    ax3.plot(square_data['t'], square_vel, 'm-', label='Square')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Speed [m/s]')
    ax3.legend()
    ax3.set_title('Speed Profiles')
    ax3.grid(True)
    
    # Acceleration comparison
    ax4 = fig.add_subplot(2, 2, 4)
    circular_acc = np.sqrt(circular_data['a_lin_x']**2 + circular_data['a_lin_y']**2 + circular_data['a_lin_z']**2)
    helix_acc = np.sqrt(helix_data['a_lin_x']**2 + helix_data['a_lin_y']**2 + helix_data['a_lin_z']**2)
    lemniscate_acc = np.sqrt(lemniscate_data['a_lin_x']**2 + lemniscate_data['a_lin_y']**2 + lemniscate_data['a_lin_z']**2)
    square_acc = np.sqrt(square_data['a_lin_x']**2 + square_data['a_lin_y']**2 + square_data['a_lin_z']**2)
    
    ax4.plot(circular_data['t'], circular_acc, 'b-', label='Circular')
    ax4.plot(helix_data['t'], helix_acc, 'r-', label='Helix')
    ax4.plot(lemniscate_data['t'], lemniscate_acc, 'g-', label='Lemniscate')
    ax4.plot(square_data['t'], square_acc, 'm-', label='Square')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Acceleration [m/s²]')
    ax4.legend()
    ax4.set_title('Acceleration Profiles')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved as: trajectory_comparison.png")


if __name__ == "__main__":
    print("Trajectory Generator Examples")
    print("=" * 40)
    
    try:
        plot_trajectories_comparison()
        print("\n✓ All example trajectories generated successfully!")
        print("\nGenerated files:")
        print("- circular_trajectory.csv")
        print("- helix_trajectory.csv") 
        print("- lemniscate_trajectory.csv")
        print("- square_trajectory.csv")
        print("- trajectory_comparison.png")
        
    except Exception as e:
        print(f"\n❌ Error generating trajectories: {e}")
        raise