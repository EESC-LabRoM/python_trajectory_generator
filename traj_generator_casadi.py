"""
Trajectory generator using CasADi for differential flatness of quadrotors.
Converted from MATLAB implementation.
"""

import casadi as ca
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def traj_generator_casadi(t, px, py, pz, psi, dt, tend, opt):
    """
    Generate quadrotor trajectory using differential flatness.
    
    Parameters:
    t : CasADi SX symbol for time
    px, py, pz : CasADi expressions for position trajectories
    psi : CasADi expression for heading angle
    dt : time step
    tend : final time
    opt : options dictionary with keys:
        - zero_lateral_overload : bool, if True no lateral overload (similar to quadrotors)
        - reset_terminal_att : bool, if True regulate terminal quaternion
    
    Returns:
    data : pandas DataFrame with trajectory data
    """
    
    print('Defining symbolics')
    
    # First derivatives (velocities)
    vx = ca.jacobian(px, t)
    vy = ca.jacobian(py, t)
    vz = ca.jacobian(pz, t)
    dpsi = ca.jacobian(psi, t)
    
    # Second derivatives (accelerations)
    ax = ca.jacobian(vx, t)
    ay = ca.jacobian(vy, t)
    az = ca.jacobian(vz, t)
    ddpsi = ca.jacobian(dpsi, t)
    
    # Third derivatives (jerks)
    jx = ca.jacobian(ax, t)
    jy = ca.jacobian(ay, t)
    jz = ca.jacobian(az, t)
    
    # Define acceleration and gravity vectors
    a = ca.vertcat(ax, ay, az)
    g = ca.vertcat(0, 0, -9.81)
    
    # Calculate thrust direction
    if opt['zero_lateral_overload']:
        e3 = a - g
    else:
        e3 = -g
    
    T = ca.norm_2(e3)
    dT = ca.jacobian(T, t)
    e3 = e3 / ca.norm_2(e3)
    
    # Define body frame
    e1 = ca.vertcat(ca.cos(psi), ca.sin(psi), 0)
    e2 = ca.cross(e3, e1)
    e2 = e2 / ca.norm_2(e2)
    e1 = ca.cross(e2, e3)
    
    # Rotation matrix
    R = ca.horzcat(e1, e2, e3)
    
    # Calculate angular velocities
    j = ca.vertcat(jx, jy, jz)
    h = (j - dT * e3) / T
    
    wx = -ca.dot(h, e2)
    wy = ca.dot(h, e1)
    wz = ca.dot(ca.vertcat(0, 0, 1), e3) * dpsi
    
    # Angular accelerations
    alphax = ca.jacobian(wx, t)
    alphay = ca.jacobian(wy, t)
    alphaz = ca.jacobian(wz, t)
    
    # Create time vector
    time = np.arange(0, tend + dt, dt)
    
    # Create CasADi functions
    fpx = ca.Function('fpx', [t], [px])
    fpy = ca.Function('fpy', [t], [py])
    fpz = ca.Function('fpz', [t], [pz])
    
    fvx = ca.Function('fvx', [t], [vx])
    fvy = ca.Function('fvy', [t], [vy])
    fvz = ca.Function('fvz', [t], [vz])
    
    fax = ca.Function('fax', [t], [ax])
    fay = ca.Function('fay', [t], [ay])
    faz = ca.Function('faz', [t], [az])
    
    fR = ca.Function('fR', [t], [R])
    
    fwx = ca.Function('fwx', [t], [wx])
    fwy = ca.Function('fwy', [t], [wy])
    fwz = ca.Function('fwz', [t], [wz])
    
    falphax = ca.Function('falphax', [t], [alphax])
    falphay = ca.Function('falphay', [t], [alphay])
    falphaz = ca.Function('falphaz', [t], [alphaz])
    
    # Evaluate functions at time points
    p_x = np.array([float(fpx(ti)) for ti in time])
    p_y = np.array([float(fpy(ti)) for ti in time])
    p_z = np.array([float(fpz(ti)) for ti in time])
    
    v_x = np.array([float(fvx(ti)) for ti in time])
    v_y = np.array([float(fvy(ti)) for ti in time])
    v_z = np.array([float(fvz(ti)) for ti in time])
    
    a_lin_x = np.array([float(fax(ti)) for ti in time])
    a_lin_y = np.array([float(fay(ti)) for ti in time])
    a_lin_z = np.array([float(faz(ti)) for ti in time])
    
    w_x = np.array([float(fwx(ti)) for ti in time])
    w_y = np.array([float(fwy(ti)) for ti in time])
    w_z = np.array([float(fwz(ti)) for ti in time])
    
    a_rot_x = np.array([float(falphax(ti)) for ti in time])
    a_rot_y = np.array([float(falphay(ti)) for ti in time])
    a_rot_z = np.array([float(falphaz(ti)) for ti in time])
    
    # Calculate quaternions
    N = len(time)
    q_w = np.zeros(N)
    q_x = np.zeros(N)
    q_y = np.zeros(N)
    q_z = np.zeros(N)
    
    for i in range(N):
        rotm = np.array(fR(time[i]))
        # Convert rotation matrix to quaternion using scipy
        r = Rotation.from_matrix(rotm.T)  # Transpose because of different convention
        quat = r.as_quat()  # [x, y, z, w] format in scipy
        q_w[i] = quat[3]  # w component
        q_x[i] = quat[0]  # x component
        q_y[i] = quat[1]  # y component
        q_z[i] = quat[2]  # z component
        
        if (i + 1) % 10 == 0:
            print(f"Generating quaternions: {(i + 1) * 100 / N:.2f} %")
    
    print('Generating table')
    
    # Set terminal velocities and accelerations to zero
    v_x[-1] = 0.0
    v_y[-1] = 0.0
    v_z[-1] = 0.0
    w_x[-1] = 0.0
    w_y[-1] = 0.0
    w_z[-1] = 0.0
    a_lin_x[-1] = 0.0
    a_lin_y[-1] = 0.0
    a_lin_z[-1] = 0.0
    
    if opt['reset_terminal_att']:
        q_w[-1] = 1.0
        q_x[-1] = 0.0
        q_y[-1] = 0.0
        q_z[-1] = 0.0
    
    # Create DataFrame
    data = pd.DataFrame({
        't': time,
        'p_x': p_x,
        'p_y': p_y,
        'p_z': p_z,
        'q_w': q_w,
        'q_x': q_x,
        'q_y': q_y,
        'q_z': q_z,
        'v_x': v_x,
        'v_y': v_y,
        'v_z': v_z,
        'w_x': w_x,
        'w_y': w_y,
        'w_z': w_z,
        'a_lin_x': a_lin_x,
        'a_lin_y': a_lin_y,
        'a_lin_z': a_lin_z,
        'a_rot_x': a_rot_x,
        'a_rot_y': a_rot_y,
        'a_rot_z': a_rot_z
    })
    
    return data