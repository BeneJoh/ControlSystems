import numpy as np
import matplotlib.pyplot as plt
import control as ct

# === Defaults for module ===
DEFAULT_TIMEPTS = np.linspace(0, 40, 2000)
DEFAULT_POS_TRAJECTORY = np.minimum(DEFAULT_TIMEPTS, 20)
DEFAULT_OVERSHOOT = 0.05
DEFAULT_POSITION_SETTLING_ERROR = 0.01  # 1 cm

# === Common Plot Helper ===
def _plot_subplots(data, labels, title=None, xlabel="Time [s]"):
    """Plot multiple time series in stacked subplots with shared X axis."""
    fig, axs = plt.subplots(len(data), 1, sharex=True, figsize=(8, 2.5 * len(data)))
    if title:
        fig.suptitle(title)
    for ax, ((T, y), label) in zip(axs, zip(data, labels)):
        ax.plot(T, y)
        ax.set_ylabel(label)
        ax.grid(True)
    axs[-1].set_xlabel(xlabel)
    plt.tight_layout()
    plt.show()

def _annotate_error_metrics(ax, T, error, tol=DEFAULT_POSITION_SETTLING_ERROR):
    """Annotate final error and settling time for error."""
    final_error = error[-1]
    
    # Settling time: last time error leaves ±tol
    idx_outside = np.where(np.abs(error) > tol)[0]
    settling_time = T[idx_outside[-1] + 1] if len(idx_outside) > 0 and idx_outside[-1] + 1 < len(T) else T[-1]
    
    # Annotation in upper right corner
    text = f"Final error: {final_error:.4f} m\nSettling time: {settling_time:.2f} s"
    ax.annotate(text,
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                ha="right", va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    
    return final_error, settling_time

# === SISO extraction ===
def siso_position(c_sys_lin):
    """Extract the SISO transfer function from force input to position output."""
    sys = ct.ss2tf(c_sys_lin, inputs=['F'], outputs=['phi', 'omega', 'x', 'v'])
    return sys[['x'], ['F']]

# === PID transfer function ===
def pid_tf(Kp, Ki, Kd, tau=1e-1):
    """Create a PID controller transfer function with optional low pass filter for D."""
    if (Kp > 0) and (Ki == 0) and (Kd == 0):
        num, den = [Kp], [1]
    elif (Ki > 0) and (Kd > 0) and (tau > 0):
        num, den = [Kp*tau + Kd, Ki*tau + Kp, Ki], [tau, 1, 0]
    elif (Kp > 0) and (Kd > 0) and (tau > 0) and (Ki == 0):
        num, den = [Kp*tau + Kd, Kp], [tau, 1]
    else:
        num, den = [Kd, Kp, Ki], [1, 0]
    return ct.tf(num, den, inputs=('e',), outputs=('u',))

# === Close-loop SISO ===
def close_siso_sys(G_p, G_c):
    """Return the closed-loop transfer function from reference to output."""
    return (G_c*G_p/(1 + G_c*G_p)).minreal()

# === PID Analysis ===


def evaluate_step_response(sys,timepts=None, overshoot_tol=DEFAULT_OVERSHOOT):
    """
    Evaluate and plot the step response performance metrics.
    
    Parameters
    ----------
    t : array_like
        Time vector of the response.
    y : array_like
        Output vector of the response.
    overshoot_tol : float, optional
        Tolerance for settling time as fraction of final value (default 5%).
    """
    timepts = DEFAULT_TIMEPTS if timepts is None else timepts

    t,y = ct.step_response(sys,timepts)
    y_final = y[-1]
    overshoot = (np.max(y) - y_final) / y_final * 100
    t_overshoot = t[np.argmax(y)]

    # Compute bounds for settling time
    upper_bound = (1 + overshoot_tol) * y_final
    lower_bound = (1 - overshoot_tol) * y_final

    # Find last time output leaves the bounds
    idx_settle = np.where((y > upper_bound) | (y < lower_bound))[0]
    settling_time = t[idx_settle[-1] + 1] if len(idx_settle) > 0 else t[-1]

    # --- Plot ---
    plt.figure()
    plt.plot(t, y, label="Step Response")
    plt.axhline(y_final, color='gray', linestyle='--', linewidth=0.8, label="Final value")
    plt.axhline(upper_bound, color='red', linestyle=':', linewidth=0.8)
    plt.axhline(lower_bound, color='red', linestyle=':', linewidth=0.8)

    # Mark overshoot
    plt.plot(t_overshoot, np.max(y), 'ro')
    plt.annotate(f"Overshoot: {overshoot:.2f}%",
                 xy=(t_overshoot, np.max(y)),
                 xytext=(t_overshoot+3, np.max(y)),
                 arrowprops=dict(arrowstyle="->"))

    # Mark settling time
    plt.axvline(settling_time, color='green', linestyle='--', linewidth=0.8)
    plt.annotate(f"Settling time: {settling_time:.2f} s",
                 xy=(settling_time, y_final),
                 xytext=(settling_time+0.5, y_final-0.1),
                 arrowprops=dict(arrowstyle="->"))

    plt.title("Step Response Evaluation")
    plt.xlabel("Time [s]")
    plt.ylabel("Output")
    plt.grid(True)
    plt.legend()
    plt.show()

    return {
        "y_out":y,
        "final_value": y_final,
        "overshoot_percent": overshoot,
        "overshoot_time": t_overshoot,
        "settling_time": settling_time
    }


def analyze_pid_control(sys, pid, timepts=None, pos_trajectory=None):
    """Simulate and plot position control response using PID."""
    timepts = DEFAULT_TIMEPTS if timepts is None else timepts
    pos_trajectory = DEFAULT_POS_TRAJECTORY if pos_trajectory is None else pos_trajectory

    closed_sys = close_siso_sys(sys, pid)
    T, y_out = ct.forced_response(closed_sys, timepts, pos_trajectory)
    _, u_out = ct.forced_response(pid, timepts, pos_trajectory - y_out)

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 10))
    plots = [
        (pos_trajectory, "Position trajectory [m]"),
        (y_out, "Position [m]"),
        (pos_trajectory - y_out, "Position error [m]"),
        (u_out, "Control input [N]")
    ]
    
    for ax, (y, label) in zip(axs, plots):
        ax.plot(T, y)
        ax.set_ylabel(label)
        ax.grid(True)
    
    # Annotate error metrics in the error plot
    _annotate_error_metrics(axs[2], T, pos_trajectory - y_out)

    axs[-1].set_xlabel("Time [s]")
    fig.suptitle("Position control with PID")
    plt.tight_layout()
    plt.show()

    return y_out, u_out


# === PID Misc States ===
def analyze_pid_misc_states(pid, c_sys_lin, error=None, timepts=None):
    """Simulate and plot other states (phi, omega, velocity) under PID control."""
    timepts = DEFAULT_TIMEPTS if timepts is None else timepts
    error = np.zeros_like(timepts) if error is None else error

    sys_tf = ct.ss2tf(c_sys_lin, inputs=['F'], outputs=['phi', 'omega', 'x', 'v'])
    states = {
        "Phi [rad]": sys_tf[['phi'], ['F']],
        "Omega [rad/s]": sys_tf[['omega'], ['F']],
        "Velocity [m/s]": sys_tf[['v'], ['F']]
    }
    data, labels = [], []
    for label, siso in states.items():
        T, y_out = ct.forced_response(siso*pid, timepts, error)
        data.append((T, y_out))
        labels.append(label)

    _plot_subplots(data, labels, title="Non-controlled states for position control with PID")

# === Controllability ===
def check_controllability(sys):
    """Check and print system controllability matrix rank."""
    Co = ct.ctrb(sys.A, sys.B)
    rank_Co = np.linalg.matrix_rank(Co)
    n = sys.A.shape[0]
    print("Controllability Matrix:\n", Co)
    print(f"Rank: {rank_Co}, System order: {n}")
    print("✅ Controllable" if rank_Co == n else "❌ Not controllable")
    return Co, rank_Co

# === LQR ===
def compute_lqr(sys, Q, R):
    """Compute LQR gain and return closed-loop system."""
    K, S, E = ct.lqr(sys, Q, R)
    _, clsys = ct.create_statefbk_iosystem(sys, K)
    return clsys, K, S, E

def resp_for_input(clsys, pos_trajectory=None, timepts=None):
    """Simulate closed-loop LQR response for a given position trajectory."""
    timepts = DEFAULT_TIMEPTS if timepts is None else timepts
    pos_trajectory = DEFAULT_POS_TRAJECTORY if pos_trajectory is None else pos_trajectory

    U = [np.zeros_like(pos_trajectory), np.zeros_like(pos_trajectory),
         pos_trajectory, np.zeros_like(pos_trajectory), np.zeros_like(pos_trajectory)]
    return ct.forced_response(clsys, timepts, U)

# === LQR Analysis ===


def analyze_lqr_control(resp_lqr, timepts=None, pos_trajectory=None):
    """Plot position tracking performance of LQR controller."""
    timepts = DEFAULT_TIMEPTS if timepts is None else timepts
    pos_trajectory = DEFAULT_POS_TRAJECTORY if pos_trajectory is None else pos_trajectory

    T = timepts
    y_out = resp_lqr.outputs['y[2]']
    f_in = resp_lqr.outputs['u[0]']

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 10))
    plots = [
        (pos_trajectory, "Position trajectory [m]"),
        (y_out, "Position [m]"),
        (pos_trajectory - y_out, "Position error [m]"),
        (f_in, "Control input [N]")
    ]
    
    for ax, (y, label) in zip(axs, plots):
        ax.plot(T, y)
        ax.set_ylabel(label)
        ax.grid(True)
    
    # Annotate error metrics in the error plot
    _annotate_error_metrics(axs[2], T, pos_trajectory - y_out)

    axs[-1].set_xlabel("Time [s]")
    fig.suptitle("Position control with LQR")
    plt.tight_layout()
    plt.show()

def analyze_lqr_misc_states(resp_lqr, timepts=None):
    """Plot additional system states (phi, omega, velocity) for LQR control."""
    timepts = DEFAULT_TIMEPTS if timepts is None else timepts
    T = timepts
    data = [
        (T, resp_lqr.outputs['y[0]']),
        (T, resp_lqr.outputs['y[1]']),
        (T, resp_lqr.outputs['y[3]'])
    ]
    labels = ["Phi [rad]", "Omega [rad/s]", "Velocity [m/s]"]

    _plot_subplots(data, labels, title="States for position control with LQR")
