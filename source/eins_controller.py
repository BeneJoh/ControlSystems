import numpy as np
import matplotlib.pyplot as plt
import control as ct

# === Defaults for module ===
DEFAULT_TIMEPTS = np.linspace(0, 40, 2000)
DEFAULT_POS_TRAJECTORY = np.minimum(DEFAULT_TIMEPTS, 20)

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
def analyze_pid_control(sys, pid, timepts=None, pos_trajectory=None):
    """Simulate and plot position control response using PID."""
    timepts = DEFAULT_TIMEPTS if timepts is None else timepts
    pos_trajectory = DEFAULT_POS_TRAJECTORY if pos_trajectory is None else pos_trajectory

    closed_sys = close_siso_sys(sys, pid)
    T, y_out = ct.forced_response(closed_sys, timepts, pos_trajectory)
    _, u_out = ct.forced_response(pid, timepts, pos_trajectory - y_out)

    _plot_subplots(
        data=[(T, pos_trajectory), (T, y_out),
              (T, pos_trajectory - y_out), (T, u_out)],
        labels=["Position trajectory [m]", "Position [m]",
                "Position error [m]", "Control input [N]"],
        title="Position control with PID"
    )
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

    _plot_subplots(
        data=[(T, pos_trajectory), (T, y_out),
              (T, pos_trajectory - y_out), (T, f_in)],
        labels=["Position trajectory [m]", "Position [m]",
                "Position error [m]", "Control input [N]"],
        title="Position control with LQR"
    )

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
