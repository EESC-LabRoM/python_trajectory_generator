# Trajectory Generator - Python Version

This program generates payload trajectories leveraging differential-flatness of quadrotors. The users need to provide algebraic equations of quadrotor flat outputs (px, py, pz, psi). Then the trajectories including pos, vel, acc, omega, alpha are generated and stored in a CSV file.

**Note**: This is a Python port of the original MATLAB implementation.

## Dependencies

- Python 3.7+ (tested with Python 3.10)
- CasADi: Symbolic computation and automatic differentiation
- NumPy: Numerical computing
- Pandas: Data manipulation and CSV output
- Matplotlib: Plotting (optional)
- SciPy: For rotation matrix to quaternion conversion

## Installation

1. Clone or download this repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install casadi numpy pandas matplotlib scipy
   ```

## Usage

### Basic Usage

1. Define parameters in `main.py`
2. Define algebraic equations of trajectories in `main.py`
3. Run the main script:
   ```bash
   python main.py
   ```

### Advanced Usage

You can also use the trajectory generator directly in your own scripts:

```python
import casadi as ca
from traj_generator_casadi import traj_generator_casadi

# Define symbolic time variable
t = ca.SX.sym('t')

# Define trajectory equations
px = 2.0 * ca.cos(0.5 * t)  # x position
py = 2.0 * ca.sin(0.5 * t)  # y position  
pz = 1.0 + 0.0 * t          # z position (constant height)
psi = 0.0 * t               # heading angle

# Set options
opt = {
    'zero_lateral_overload': True,   # No lateral overload (quadrotor-like)
    'reset_terminal_att': False      # Don't reset terminal attitude
}

# Generate trajectory
data = traj_generator_casadi(t, px, py, pz, psi, dt=0.1, tend=20.0, opt=opt)

# Save to CSV
data.to_csv('my_trajectory.csv', index=False)
```

## Options

The `opt` dictionary supports the following parameters:

- `zero_lateral_overload` (bool): 
  - `True`: No lateral overload, similar to quadrotors
  - `False`: Zero pitch and roll
- `reset_terminal_att` (bool):
  - `True`: Regulate terminal quaternion to identity
  - `False`: Let terminal attitude evolve naturally

## Output Format

The generated CSV file contains the following columns:

| Column    | Description                    | Units      |
|-----------|--------------------------------|------------|
| t         | Time                          | s          |
| p_x, p_y, p_z | Position                  | m          |
| q_w, q_x, q_y, q_z | Quaternion (w,x,y,z)  | -          |
| v_x, v_y, v_z | Velocity                  | m/s        |
| w_x, w_y, w_z | Angular velocity          | rad/s      |
| a_lin_x, a_lin_y, a_lin_z | Linear acceleration | m/s²   |
| a_rot_x, a_rot_y, a_rot_z | Angular acceleration | rad/s² |

## Example Trajectories

### Figure-Eight Trajectory (Default)

The default configuration generates a figure-eight trajectory:

```python
# Time distortion for smooth start/stop
k = 1
t0 = 5
t1 = tend - 5.0
ts = ca.log(1 + ca.exp(k * (t - t0))) - ca.log(1 + ca.exp(k * (t - t1)))

# Figure-eight equations
px = 2.5 * ca.cos(0.25 * ts) + 0.2
py = 2.0 * ca.sin(0.5 * ts)
pz = 1.0  # constant height
psi = 0.25 * ts  # gradually changing heading
```

### Circular Trajectory

```python
px = 2.0 * ca.cos(0.5 * t)
py = 2.0 * ca.sin(0.5 * t)
pz = 1.0 + 0.0 * t
psi = 0.0 * t
```

### Helix Trajectory

```python
px = 2.0 * ca.cos(0.5 * t)
py = 2.0 * ca.sin(0.5 * t)
pz = 1.0 + 0.1 * t
psi = 0.2 * t
```

## Testing

Run the test suite to validate the implementation:

```bash
python test_trajectory.py
```

The test script validates:
- Basic trajectory generation functionality
- Figure-eight trajectory generation
- Different option configurations
- Output format and data integrity

## Differences from MATLAB Version

This Python implementation maintains full compatibility with the MATLAB version while offering these improvements:

1. **Cross-platform compatibility**: No MATLAB license required
2. **Modern Python ecosystem**: Easy integration with other Python tools
3. **Virtual environment support**: Clean dependency management
4. **Automated testing**: Built-in test suite for validation
5. **Type hints and documentation**: Better code maintainability

## Performance Notes

- Quaternion generation is the most computationally expensive part
- Progress indicators are shown during quaternion calculation
- For large trajectories (high resolution or long duration), consider reducing the time resolution or using smaller time windows

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed in your Python environment
2. **CasADi installation**: On some systems, you might need to install CasADi from source
3. **Plotting issues**: If running in a headless environment, set `make_plot = False` in `main.py`
4. **Memory issues**: For very long trajectories, consider processing in smaller chunks

### Getting Help

If you encounter issues:
1. Check that your trajectory equations are valid CasADi expressions
2. Verify that your time variable `t` is properly defined as `ca.SX.sym('t')`
3. Ensure all required dependencies are installed
4. Run the test suite to validate your installation

## License

Same as the original MATLAB implementation.