"""
Test script for the Python trajectory generator.
Validates the implementation and runs basic functionality tests.
"""

import numpy as np
import casadi as ca
import pandas as pd
from traj_generator_casadi import traj_generator_casadi
import matplotlib.pyplot as plt


def test_basic_trajectory():
    """Test basic trajectory generation functionality."""
    
    print("Testing basic trajectory generation...")
    
    # Simple test parameters
    dt = 0.2
    tend = 10.0
    
    opt = {
        'zero_lateral_overload': True,
        'reset_terminal_att': False
    }
    
    # Define symbolic time
    t = ca.SX.sym('t')
    
    # Simple circular trajectory
    px = 2.0 * ca.cos(0.5 * t)
    py = 2.0 * ca.sin(0.5 * t)
    pz = 1.0 + 0.0 * t  # constant height
    psi = 0.0 * t  # zero heading
    
    try:
        # Generate trajectory
        data = traj_generator_casadi(t, px, py, pz, psi, dt, tend, opt)
        
        # Basic validation checks
        assert isinstance(data, pd.DataFrame), "Output should be a pandas DataFrame"
        assert len(data) > 0, "DataFrame should not be empty"
        
        # Check required columns
        required_cols = ['t', 'p_x', 'p_y', 'p_z', 'q_w', 'q_x', 'q_y', 'q_z',
                        'v_x', 'v_y', 'v_z', 'w_x', 'w_y', 'w_z',
                        'a_lin_x', 'a_lin_y', 'a_lin_z', 'a_rot_x', 'a_rot_y', 'a_rot_z']
        
        for col in required_cols:
            assert col in data.columns, f"Missing column: {col}"
        
        # Check time vector
        expected_time_points = int(tend / dt) + 1
        assert len(data) == expected_time_points, f"Expected {expected_time_points} time points, got {len(data)}"
        
        # Check that quaternions are normalized
        quat_norm = np.sqrt(data['q_w']**2 + data['q_x']**2 + data['q_y']**2 + data['q_z']**2)
        assert np.allclose(quat_norm, 1.0, atol=1e-6), "Quaternions should be normalized"
        
        # Check terminal conditions
        assert abs(data['v_x'].iloc[-1]) < 1e-10, "Terminal velocity x should be zero"
        assert abs(data['v_y'].iloc[-1]) < 1e-10, "Terminal velocity y should be zero"
        assert abs(data['v_z'].iloc[-1]) < 1e-10, "Terminal velocity z should be zero"
        
        print("✓ Basic trajectory generation test passed!")
        return data
        
    except Exception as e:
        print(f"✗ Basic trajectory test failed: {e}")
        raise


def test_figure_eight_trajectory():
    """Test the figure-eight trajectory from the original main script."""
    
    print("Testing figure-eight trajectory...")
    
    # Parameters from original main.m
    dt = 0.1
    tend = 40.0
    
    opt = {
        'zero_lateral_overload': True,
        'reset_terminal_att': False
    }
    
    # Define symbolic time
    t = ca.SX.sym('t')
    
    # Time distortion for smooth start (from original)
    k = 1
    t0 = 5
    t1 = tend - 5.0
    ts = ca.log(1 + ca.exp(k * (t - t0))) - ca.log(1 + ca.exp(k * (t - t1)))
    zvars = 1e-15 * ca.sin(ts)
    
    # Figure-eight trajectory (from original)
    px = 2.5 * ca.cos(0.25 * ts) + 0.2
    py = 2.0 * ca.sin(0.5 * ts)
    pz = zvars + 1
    psi = 0.25 * ts
    
    try:
        # Generate trajectory
        data = traj_generator_casadi(t, px, py, pz, psi, dt, tend, opt)
        
        # Validate trajectory characteristics
        assert len(data) > 0, "DataFrame should not be empty"
        
        # Check that we have a reasonable trajectory (not checking strict periodicity)
        start_pos = np.array([data['p_x'].iloc[0], data['p_y'].iloc[0]])
        end_pos = np.array([data['p_x'].iloc[-1], data['p_y'].iloc[-1]])
        distance_to_start = np.linalg.norm(end_pos - start_pos)
        
        # Just check that trajectory doesn't explode
        assert distance_to_start < 10.0, f"Trajectory should be reasonable, distance from start: {distance_to_start}"
        
        # Check that trajectory actually moves (not stuck at origin)
        max_distance_from_origin = np.max(np.sqrt(data['p_x']**2 + data['p_y']**2))
        assert max_distance_from_origin > 1.0, "Trajectory should move away from origin"
        
        # Check altitude stays roughly constant
        z_variation = data['p_z'].max() - data['p_z'].min()
        assert z_variation < 0.1, f"Altitude should be roughly constant, variation: {z_variation}"
        
        print("✓ Figure-eight trajectory test passed!")
        return data
        
    except Exception as e:
        print(f"✗ Figure-eight trajectory test failed: {e}")
        raise


def test_with_different_options():
    """Test with different option settings."""
    
    print("Testing different option configurations...")
    
    dt = 0.2
    tend = 5.0
    
    # Test with zero lateral overload disabled
    opt1 = {
        'zero_lateral_overload': False,
        'reset_terminal_att': False
    }
    
    # Test with terminal attitude reset
    opt2 = {
        'zero_lateral_overload': True,
        'reset_terminal_att': True
    }
    
    t = ca.SX.sym('t')
    px = ca.cos(t)
    py = ca.sin(t)
    pz = 1.0 + 0.0 * t
    psi = 0.0 * t
    
    try:
        # Test both configurations
        data1 = traj_generator_casadi(t, px, py, pz, psi, dt, tend, opt1)
        data2 = traj_generator_casadi(t, px, py, pz, psi, dt, tend, opt2)
        
        # Check terminal attitude reset
        if opt2['reset_terminal_att']:
            assert abs(data2['q_w'].iloc[-1] - 1.0) < 1e-10, "Terminal quaternion w should be 1"
            assert abs(data2['q_x'].iloc[-1]) < 1e-10, "Terminal quaternion x should be 0"
            assert abs(data2['q_y'].iloc[-1]) < 1e-10, "Terminal quaternion y should be 0"
            assert abs(data2['q_z'].iloc[-1]) < 1e-10, "Terminal quaternion z should be 0"
        
        print("✓ Different options test passed!")
        
    except Exception as e:
        print(f"✗ Different options test failed: {e}")
        raise


def run_all_tests():
    """Run all test functions."""
    
    print("Running trajectory generator tests...\n")
    
    try:
        # Run tests
        test_basic_trajectory()
        test_figure_eight_trajectory()
        test_with_different_options()
        
        print("\n🎉 All tests passed successfully!")
        
        # Generate a sample trajectory for visual inspection
        print("\nGenerating sample trajectory for visual inspection...")
        return generate_sample_trajectory()
        
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")
        raise


def generate_sample_trajectory():
    """Generate a sample trajectory for visual inspection."""
    
    dt = 0.1
    tend = 20.0
    
    opt = {
        'zero_lateral_overload': True,
        'reset_terminal_att': False
    }
    
    t = ca.SX.sym('t')
    
    # Interesting 3D trajectory
    px = 3.0 * ca.cos(0.3 * t)
    py = 2.0 * ca.sin(0.6 * t)
    pz = 1.0 + 0.5 * ca.sin(0.2 * t)
    psi = 0.1 * t
    
    data = traj_generator_casadi(t, px, py, pz, psi, dt, tend, opt)
    
    # Save test trajectory
    data.to_csv('test_trajectory.csv', index=False)
    print("Sample trajectory saved to: test_trajectory.csv")
    
    return data


if __name__ == "__main__":
    data = run_all_tests()