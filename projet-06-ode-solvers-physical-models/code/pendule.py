import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.colors import LogNorm
from joblib import Parallel, delayed  
from scipy.integrate import solve_ivp
import time
import partie1

# ———————————————————— Simple pendulum ————————————————————

def simple_pendulum(t, state, gravity=9.81, length=1.0):
    """
    Defines the system of equations for a simple pendulum.

    Params:
        t (float) : current time 
        state (array of float) : [angle, angular velocity] at time t
        gravity (float) : acceleration due to gravity (default 9.81 m/s^2)
        length (float) : length of the pendulum (default 1.0 m)

    Returns:
        (array of float) : [angular velocity, angular acceleration] at time t
    """
    angle, omega = state
    return np.array([omega, - (gravity/length) * np.sin(angle)])

def pendulum_period(initial_state, t_start, t_end, ode_func, integrator, dt=0.001):
    """
    Computes the average period of a simple pendulum by calculating the time between consecutive peaks
    in the angle and averaging the results.

    Params:
        initial_state (array of float) : initial state [angle, angular velocity]
        t_start (float) : starting time for integration
        t_end (float) : ending time for integration
        ode_func (function) : the ODE function to integrate
        integrator (function) : the integration method (Euler, Midpoint, Heun, Runge-Kutta)
        dt (float) : time step for integration (default 0.001)

    Returns:
        float or None : the average period of the pendulum (time between consecutive peaks), or None if unable to compute
    """
    # Calculate the number of time steps
    N = int((t_end - t_start) / dt)
    # Use the integrator to solve the ODE over N steps
    time_vals, state_vals = partie1.meth_n_step(initial_state, t_start, N, dt, ode_func, integrator)
    # Extract the angles from the solution
    angles = state_vals[:,0]
    # Find the peaks (the times when the pendulum reaches maximum displacement)
    peaks, _ = find_peaks(angles)
    
    # Check if we have enough peaks to calculate periods
    if len(peaks) >= 2:
        periods = []
        for i in range(1, len(peaks)):
            t0, t1 = time_vals[peaks[i-1]], time_vals[peaks[i]]
            periods.append(t1 - t0)  # Time between each consecutive peak
        
        # Return the average period
        return np.mean(periods)  # Compute and return the average of all periods
    
    return None

def plot_pendulum_frequency(step_method, length=1.0, gravity=9.81):
    """
    Plots the frequency of a simple pendulum as a function of the initial angle.

    Params:
        step_method (function) : the integration method (Euler, Midpoint, Heun, Runge-Kutta)
        length (float) : length of the pendulum
        gravity (float) : acceleration due to gravity (default 9.81 m/s^2)

    Returns:
        None : plots the frequency as a function of the initial angle θ0
    """
    # Generate an array of initial angles from 0.01 to π-0.01 radians
    th0 = np.linspace(0.01, np.pi-0.01, 50)
    freq = []
    for θ in th0:
        # For each initial angle, solve for the pendulum period
        state0 = np.array([θ, 0.0])  # Initial condition: angle and initial velocity (zero)
        T = pendulum_period(
            state0, 0, 10,  # Time range from 0 to 10 seconds
            lambda t, y: simple_pendulum(t, y, gravity, length),  # Pendulum ODE
            step_method,  # Integration method to use
            dt=0.001  # Time step
        )
        # If a valid period T is found, compute the frequency (1/T)
        freq.append(1/T if T and T > 0 else np.nan)

    # Plot the frequency vs initial angle θ0
    plt.plot(th0, freq, label="Measured")
    # The theoretical frequency for small angles is sqrt(g/l)/(2 * pi) (this is an approximation valid for small θ0)
    plt.axhline(np.sqrt(gravity/length) / (2*np.pi), color='r', ls='--', label="Small angle freq (Hz)")
    plt.xlabel("θ0 (rad)")  # Label for the x-axis (initial angle in radians)
    plt.ylabel("Frequency (Hz)")  # Label for the y-axis (frequency in Hz)
    plt.legend()  # Show legend for the plot
    plt.grid()  # Add grid to the plot for better visibility
    plt.title("Pendulum Frequency vs θ0")  # Title of the plot
    plt.show()  # Display the plot


# ———————————————————— Double pendulum ————————————————————

def double_pendulum(t, state, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
    """
    Defines the equations of motion for a double pendulum system.

    Params:
        t (float): current time (not directly used, required by ODE solver)
        state (array of float): [theta1, omega1, theta2, omega2], angles and angular velocities
        m1, m2 (float): masses of the first and second pendulum (default 1.0)
        l1, l2 (float): lengths of the first and second pendulum arms (default 1.0)
        g (float): gravitational acceleration (default 9.81 m/s^2)

    Returns:
        array of float: [dtheta1/dt, domega1/dt, dtheta2/dt, domega2/dt], time derivatives
    """
    if not np.all(np.isfinite(state)):
        return np.zeros_like(state)

    theta1, omega1, theta2, omega2 = state
    delta = theta2 - theta1

    # Angular velocities
    dtheta1 = omega1
    dtheta2 = omega2

    # Trigonometric precomputations
    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)

    # Compute acceleration of first pendulum
    num1 = g * (np.sin(theta2) * cos_delta - np.sin(theta1)) + \
           (l2 * omega2**2 + l1 * omega1**2 * cos_delta) * sin_delta
    den1 = l1 * (1 - m2 / (m1 + m2) * cos_delta**2)

    if abs(den1) < 1e-10:
        domega1 = 0
    else:
        domega1 = num1 / den1

    # Compute acceleration of second pendulum
    num2 = -g * (np.sin(theta2) - np.sin(theta1) * cos_delta) + \
           (l1 * (m1 + m2) * omega1**2 + l2 * m2 * omega2**2 * cos_delta) * sin_delta
    den2 = l2 * (m1 + m2 * sin_delta**2)

    if abs(den2) < 1e-10:
        domega2 = 0
    else:
        domega2 = num2 / den2

    # Clip values to avoid divergence in case of instabilities
    domega1 = np.clip(domega1, -1e6, 1e6)
    domega2 = np.clip(domega2, -1e6, 1e6)

    return np.array([dtheta1, domega1, dtheta2, domega2])

def get_positions(state, l1, l2):
    """
    Computes the Cartesian coordinates of both pendulum masses from the angular state.

    Params:
        state (array of float): [theta1, omega1, theta2, omega2]
        l1, l2 (float): lengths of the pendulum arms

    Returns:
        tuple of float: (x1, y1, x2, y2) positions of the first and second masses
    """
    theta1, _, theta2, _ = state

    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)

    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)

    return x1, y1, x2, y2

def double_pendulum_motion(s1, t0, t1, dt, m1, m2, l1, l2, g, integrator):
    """
    Simulates and plots the motion of a double pendulum.

    Params:
        s1 (array): Initial state [theta1, omega1, theta2, omega2], where theta1, omega1 are the 
                    angle and angular velocity of the first pendulum, and theta2, omega2 are 
                    for the second pendulum.
        t0 (float): Initial time of the simulation.
        t1 (float): Final time of the simulation.
        dt (float): Time step for the simulation.
        m1 (float): Mass of the first pendulum (default 1.0).
        m2 (float): Mass of the second pendulum (default 1.0).
        l1 (float): Length of the first pendulum (default 1.0).
        l2 (float): Length of the second pendulum (default 1.0).
        g (float): Gravitational acceleration (default 9.81 m/s^2).
        integrator (function): Integration method used to solve the ODE (e.g., Euler, Runge-Kutta).

    Returns:
        None: The function plots the motion of the double pendulum.
    """
    # Number of time steps based on time range and time step (dt)
    N = int((t1 - t0) / dt)
    
    # Define the ODE system for the double pendulum
    ode = lambda t, y: double_pendulum(t, y, m1, m2, l1, l2, g)
    
    # Solve the ODE using the integrator function
    time_vals, state_vals = partie1.meth_n_step(s1, t0, N, dt, ode, integrator)
    
    # Extract the positions of the two pendulum masses at each time step
    positions = [get_positions(state, l1, l2) for state in state_vals]
    x1, y1, x2, y2 = zip(*positions)
    
    # Create a plot for the trajectories
    plt.figure(figsize=(12, 8))
    
    # Plot the trajectories of both pendulum masses
    plt.plot(x1, y1, label="Mass 1 Trajectory", color="green", alpha=0.7, linewidth=1.5)
    plt.plot(x2, y2, label="Mass 2 Trajectory", color="blue", alpha=0.7, linewidth=1.5)
    
    # Plot the rods connecting the masses at their initial positions
    plt.plot([0, x1[0]], [0, y1[0]], 'k-', lw=1.5)  # Rod 1
    plt.plot([x1[0], x2[0]], [y1[0], y2[0]], 'k-', lw=1.5)  # Rod 2
    
    # Plot the initial positions of the masses
    plt.plot(0, 0, 'ko', ms=10)  # Pivot point (fixed)
    plt.plot(x1[0], y1[0], 'go', ms=10*np.sqrt(m1))  # Mass 1 (scaled by its mass)
    plt.plot(x2[0], y2[0], 'bo', ms=10*np.sqrt(m2))  # Mass 2 (scaled by its mass)
    
    # Set the labels and title for the plot
    plt.xlabel("X Position (m)", fontsize=14)
    plt.ylabel("Y Position (m)", fontsize=14)
    plt.title("Double Pendulum Trajectory", fontsize=16)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Set the plot aspect ratio to be equal (so that the motion looks correct)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Display the plot to show the motion of the double pendulum
    plt.show()


def double_pendulum_flip_map(m1, m2, l1, l2, g, N, h, grid_size=100, n_jobs=-1):
    """
    Generates a flip-time map for the double pendulum.

    This function computes the time it takes for the pendulum to "flip"
    (i.e., one of the angles exceeds ±π) for a grid of initial angles θ1 and θ2.
    The result is displayed as a heatmap.

    Parameters:
    - m1, m2: masses of the two pendulum arms
    - l1, l2: lengths of the two arms
    - g: gravitational acceleration
    - N: number of steps in the simulation
    - h: time step
    - grid_size: resolution of the θ1 and θ2 grid (default: 100)
    - n_jobs: number of parallel jobs for computation (default: -1 = all CPUs)
    """

    theta1_vals = np.linspace(-np.pi, np.pi, grid_size)  # Grid of θ1 values
    theta2_vals = np.linspace(-np.pi, np.pi, grid_size)  # Grid of θ2 values

    # # Detects the first flip using RK4 integrator (slow but gives us same result)
    # def check_first_flip(y0, tmax):
    #     """
    #     Checks for the first flip using custom integration method.
        
    #     Params:
    #         y0 (array): Initial state [θ1, ω1, θ2, ω2]
    #         tmax (float): Maximum simulation time
            
    #     Returns:
    #         float: Time of first flip or infinity if no flip occurs
    #     """
    #     # Define the ODE system for the double pendulum
    #     ode_func = lambda t, y: double_pendulum(t, y, m1, m2, l1, l2, g)
        
    #     # Calculate number of steps needed
    #     num_steps = int(tmax / h)
        
    #     ts, ys = partie1.meth_n_step(y0, t0=0, N=num_steps, h=h, f=ode_func, step_method=partie1.step_rk4)
        
    #     # Check for flip condition at each time step
    #     for k in range(0, len(ts), 10):
    #         theta1, theta2 = ys[k, 0], ys[k, 2]
    #         if abs(theta1) >= np.pi or abs(theta2) >= np.pi:
    #             return ts[k]  # Return first flip time
        
    #     return np.inf  # No flip occurred within the time window

    # Detects the first flip using an ODE scipy solver (faster because we wanted images with bigger resolution)
    def check_first_flip(y0, tmax):
        def event_flip(t, y):
            # Triggers when |θ1| or |θ2| exceeds π
            return max(abs(y[0]), abs(y[2])) - np.pi
        event_flip.terminal = True  # Stop integration at the first flip
        event_flip.direction = 0    # Trigger in any direction

        sol = solve_ivp(
            lambda t, y: double_pendulum(t, y, m1, m2, l1, l2, g),
            [0, tmax],
            y0,
            method='RK45',
            events=event_flip,
            rtol=1e-3,
            atol=1e-6
        )
        # If a flip occurred, return the flip time
        if sol.t_events[0].size > 0:
            return sol.t_events[0][0]
        return np.inf  # No flip occurred within the time window

    # Computes the flip time for a given pair of initial angles (θ1, θ2)
    def compute_flip_time(i, j):
        # Skip initial conditions with high potential energy (stable equilibrium)
        if (3 * np.cos(theta1_vals[i]) + np.cos(theta2_vals[j])) > 2:
            return np.inf
        y0 = [theta1_vals[i], 0, theta2_vals[j], 0]  # Initial state: angles and zero velocities
        return check_first_flip(y0, N * h)

    # Run the flip-time computation in parallel for all grid points
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_flip_time)(i, j)
        for i in range(grid_size)
        for j in range(grid_size)
    )
    flip_times = np.array(results).reshape(grid_size, grid_size)

    # Plot the flip-time map
    plt.figure(figsize=(12, 10))
    X, Y = np.meshgrid(theta1_vals, theta2_vals)
    cs = plt.contourf(X, Y, flip_times.T, cmap='hot')  # Transpose so θ1 is x-axis, θ2 is y-axis

    plt.colorbar(cs, label='Flip time (s)')
    plt.xlabel('Initial θ₁ (rad)')
    plt.ylabel('Initial θ₂ (rad)')
    plt.title(f'Flip Time Map - gridSize: {grid_size}, N={N}, h={h}')
    plt.tight_layout()
    plt.show()

# ----------------------- Main Execution -----------------------
if __name__ == "__main__":
    # Parameters
    m1 = m2 = 1.0      # Masses of the two pendulum arms 
    l1 = l2 = 1.0      # Lengths of the pendulum arms 
    g = 9.81           # Gravitational acceleration 
    t0 = 0.0           # Start time of the simulation 
    t1 = 10.0           # End time of the simulation 
    dt = 0.01          # Time step for the simulation
    
    # Simple pendulum frequency
    plot_pendulum_frequency(partie1.step_rk4)
    
    # Initial state for the double pendulum
    s1 = np.array([np.pi/2, 0.0, np.pi/2 + 0.1, 0.0])
    
    # Simulate and visualize the motion of the double pendulum
    double_pendulum_motion(s1, t0, t1, dt, m1, m2, l1, l2, g, partie1.step_rk4)
    
    # Generate and display the flip-time map of the double pendulum
    double_pendulum_flip_map(m1, m2, l1, l2, g, 1500, 0.01, grid_size=100, n_jobs=-1)
    
