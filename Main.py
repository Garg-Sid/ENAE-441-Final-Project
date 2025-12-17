import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Constants and station data used throughout Part 1
MU_EARTH_KM3_S2 = 398600.4418  # km^3/s^2
EARTH_RADIUS_KM = 6378.137
EARTH_ROTATION_RATE = 7.292115e-5  # rad/s
LOCAL_SIDEREAL_TIME_0 = 0.0  # rad
DEG2RAD = np.pi / 180.0

DSN_SITES: Dict[int, Dict[str, float]] = {
    0: {"name": "Goldstone", "lat": 35.297 * DEG2RAD, "lon": -116.914 * DEG2RAD},
    1: {"name": "Madrid", "lat": 40.4311 * DEG2RAD, "lon": -4.248 * DEG2RAD},
    2: {"name": "Canberra", "lat": -35.4023 * DEG2RAD, "lon": 148.9813 * DEG2RAD},
}



def load_numpy_data(file_path: str) -> np.ndarray:
    """Load a numpy file relative to this script."""
    cur_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    data = np.load(cur_dir + file_path, allow_pickle=True)
    print(f"Loaded data from {file_path}")
    return data


# ---- Part 1a: Nonlinear dynamics and measurement functions ----
def continuous_dynamics(_t: float, state: np.ndarray) -> np.ndarray:
    """f(X): Two-body orbital dynamics with state [r, v]."""
    r = state[:3]
    v = state[3:]
    r_norm = np.linalg.norm(r)
    if r_norm == 0:
        raise ValueError("Position norm is zero; dynamics undefined.")
    accel = -MU_EARTH_KM3_S2 * r / r_norm**3
    return np.hstack((v, accel))


def site_state_inertial(station_index: int, t: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return inertial position and velocity of the ground station at time t."""
    site = DSN_SITES[station_index]
    lat = site["lat"]
    lon = site["lon"]
    theta = lon + EARTH_ROTATION_RATE * t + LOCAL_SIDEREAL_TIME_0
    cos_lat = np.cos(lat)
    r_site = EARTH_RADIUS_KM * np.array(
        [cos_lat * np.cos(theta), cos_lat * np.sin(theta), np.sin(lat)]
    )
    omega_vec = np.array([0.0, 0.0, EARTH_ROTATION_RATE])
    v_site = np.cross(omega_vec, r_site)
    return r_site, v_site


def measurement_function(state: np.ndarray, t: float, station_index: int) -> np.ndarray:
    """h(X): Range and range-rate between spacecraft and station."""
    r = state[:3]
    v = state[3:]
    r_site, v_site = site_state_inertial(station_index, t)
    rho_vec = r - r_site
    rho = np.linalg.norm(rho_vec)
    if rho == 0:
        raise ValueError("Spacecraft is collocated with the ground station.")
    rho_dot = np.dot(rho_vec, v - v_site) / rho
    return np.array([rho, rho_dot])


# ---- Part 1b: Linearized dynamics and measurement matrices ----
def dynamics_jacobian(state: np.ndarray) -> np.ndarray:
    """A(t): Jacobian of dynamics about the current state."""
    r = state[:3]
    r_norm = np.linalg.norm(r)
    if r_norm == 0:
        raise ValueError("Position norm is zero; Jacobian undefined.")
    I3 = np.eye(3)
    r_outer = np.outer(r, r)
    dadr = -MU_EARTH_KM3_S2 * (I3 / r_norm**3 - 3.0 * r_outer / r_norm**5)
    top = np.hstack((np.zeros((3, 3)), I3))
    bottom = np.hstack((dadr, np.zeros((3, 3))))
    return np.vstack((top, bottom))


def measurement_jacobian(state: np.ndarray, t: float, station_index: int) -> np.ndarray:
    """C(t): Jacobian of the measurement function."""
    r = state[:3]
    v = state[3:]
    r_site, v_site = site_state_inertial(station_index, t)
    rho_vec = r - r_site
    rho = np.linalg.norm(rho_vec)
    if rho == 0:
        raise ValueError("Spacecraft is collocated with the ground station.")
    drho_dr = rho_vec / rho
    drho_dv = np.zeros(3)
    v_rel = v - v_site
    rho_dot = np.dot(rho_vec, v_rel) / rho
    drhodot_dr = v_rel / rho - (rho_dot / rho**2) * rho_vec
    drhodot_dv = rho_vec / rho
    H = np.zeros((2, 6))
    H[0, :3] = drho_dr
    H[0, 3:] = drho_dv
    H[1, :3] = drhodot_dr
    H[1, 3:] = drhodot_dv
    return H


# ---- Part 1c: Discretization utilities ----
def discrete_state_transition(t0: float, t1: float, state0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the dynamics and variational equations to get Φ(t1,t0)."""
    def augmented_ode(t, y):
        x = y[:6]
        phi = y[6:].reshape(6, 6)
        x_dot = continuous_dynamics(t, x)
        phi_dot = dynamics_jacobian(x) @ phi
        return np.hstack((x_dot, phi_dot.flatten()))

    y0 = np.hstack((state0, np.eye(6).flatten()))
    result = solve_ivp(
        augmented_ode, (t0, t1), y0, rtol=1e-9, atol=1e-9, dense_output=False
    )
    if not result.success:
        raise RuntimeError("State transition integration failed.")
    state1 = result.y[:6, -1]
    phi = result.y[6:, -1].reshape(6, 6)
    return state1, phi


def discrete_measurement_matrix(state: np.ndarray, t: float, station_index: int) -> np.ndarray:
    """Hk evaluated at (t, state)."""
    return measurement_jacobian(state, t, station_index)


# ---- Part 1d: Noise matrices ----
def process_noise_matrix(dt: float, accel_noise_std: float = 1e-6) -> np.ndarray:
    """
    Qk derived from continuous white acceleration noise.
    accel_noise_std is in km/s^2; Q has units consistent with state [km, km/s].
    """
    q = accel_noise_std**2
    I3 = np.eye(3)
    Q_rr = (dt**3 / 3.0) * I3
    Q_rv = (dt**2 / 2.0) * I3
    Q_vr = Q_rv
    Q_vv = dt * I3
    top = np.hstack((Q_rr, Q_rv))
    bottom = np.hstack((Q_vr, Q_vv))
    return q * np.vstack((top, bottom))


def measurement_noise_matrix() -> np.ndarray:
    """Rk using 1 m std-dev in range and 1 cm/s in range-rate expressed in km units."""
    sigma_r_km = 1e-3
    sigma_rdot_km_s = 1e-5
    return np.diag([sigma_r_km**2, sigma_rdot_km_s**2])


# ---- Part 1e: Plotting the measurements ----
def part1e_plot_measurements(data: Optional[np.ndarray] = None) -> plt.Figure:
    """Scatter plot of range and range-rate grouped by DSN site."""
    if data is None:
        data = load_numpy_data("Project-Measurements-Easy.npy")
    t = data[:, 0]
    idx = data[:, 1].astype(int)
    rho = data[:, 2]
    rhodot = data[:, 3]

    fig = plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(2, 1, 1)
    for station in np.unique(idx):
        mask = idx == station
        ax1.plot(t[mask], rho[mask], ".", label=f"DSN {station}")
    ax1.set_ylabel("Range ρ (km)")
    ax1.set_title("Range / Range-rate Measurements vs Time")
    ax1.grid(True)
    ax1.legend()

    ax2 = plt.subplot(2, 1, 2)
    for station in np.unique(idx):
        mask = idx == station
        ax2.plot(t[mask], rhodot[mask], ".", label=f"DSN {station}")
    ax2.set_xlabel("Time t (s)")
    ax2.set_ylabel("Range-rate ρ̇ (km/s)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    return fig


# ---- Part 2: EKF pseudocode ----
EKF_PSEUDOCODE = """
Extended Kalman Filter Pseudocode (per-measurement loop)
Inputs: dynamics f(.), measurement h(.), Jacobians A(.), H(.), process noise Qk, measurement noise Rk,
        initial state mean x0 and covariance P0, measurements {Yk, tk, station_k}.

for each measurement index k:
    # Prediction (time update)
    Propagate state mean using nonlinear dynamics from t_{k-1} to tk:
        x_minus = PropagateState(x_plus_previous, t_{k-1}, tk)
    Integrate state-transition matrix Φ(tk, t_{k-1}) alongside propagation.
    Assemble discrete process noise Qk for the elapsed time Δt.
    Propagate covariance:
        P_minus = Φ P_plus_previous Φ^T + Qk

    # Measurement Update (skip if measurement unavailable)
    Evaluate measurement prediction:
        y_hat = h(x_minus, tk, station_k)
    Compute measurement Jacobian:
        Hk = ∂h/∂x |_{x_minus}
    Innovation covariance:
        Sk = Hk P_minus Hk^T + Rk
    Kalman gain:
        Kk = P_minus Hk^T Sk^{-1}
    Innovation:
        νk = Yk - y_hat
    Updated state:
        x_plus = x_minus + Kk νk
    Updated covariance:
        P_plus = (I - Kk Hk) P_minus (I - Kk Hk)^T + Kk Rk Kk^T   # numerically stable Joseph form

Return stored {x_minus, P_minus, x_plus, P_plus} for analysis.
"""


def print_part2_pseudocode() -> None:
    """Display the EKF plan that grading staff can reference."""
    print(EKF_PSEUDOCODE.strip())


# ---- Helper: Keplerian elements to Cartesian state ----
def rotation_matrix_1(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation_matrix_3(angle: float) -> np.ndarray:
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def coe_to_cartesian_state(
    a: float, e: float, inc: float, arg_perigee: float, raan: float, true_anomaly: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert classical orbital elements (km, rad) to ECI position/velocity."""
    p = a * (1.0 - e**2)
    r_pf = (p / (1.0 + e * np.cos(true_anomaly))) * np.array(
        [np.cos(true_anomaly), np.sin(true_anomaly), 0.0]
    )
    v_pf = np.sqrt(MU_EARTH_KM3_S2 / p) * np.array(
        [-np.sin(true_anomaly), e + np.cos(true_anomaly), 0.0]
    )
    rotation = rotation_matrix_3(raan) @ rotation_matrix_1(inc) @ rotation_matrix_3(arg_perigee)
    r_eci = rotation @ r_pf
    v_eci = rotation @ v_pf
    return r_eci, v_eci


# ---- Part 3: Pure prediction implementation ----
def part3_initial_conditions() -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Provide x0, P0, process noise tuning, and R.
    Position sigma of 10 km and velocity sigma of 0.01 km/s reflect launch dispersions.
    Process noise uses 1e-5 km/s^2 (unmodeled accelerations), R from Part 1d.
    """
    a = 7000.0
    e = 0.2
    inc = 45.0 * DEG2RAD
    arg_perigee = 0.0 * DEG2RAD
    raan = 270.0 * DEG2RAD
    true_anomaly = 78.75 * DEG2RAD
    r0, v0 = coe_to_cartesian_state(a, e, inc, arg_perigee, raan, true_anomaly)
    x0 = np.hstack((r0, v0))
    pos_sigma = 10.0  # km
    vel_sigma = 0.01  # km/s
    P0 = np.diag(
        [pos_sigma**2, pos_sigma**2, pos_sigma**2, vel_sigma**2, vel_sigma**2, vel_sigma**2]
    )
    accel_noise_std = 1e-6  # km/s^2
    R = measurement_noise_matrix()
    return x0, P0, accel_noise_std, R

def elevation_angle(state: np.ndarray, t: float, station_index: int) -> float:
    r = state[:3]
    r_site, _ = site_state_inertial(station_index, t)
    rho = r - r_site
    rho_hat = rho / np.linalg.norm(rho)
    zenith_hat = r_site / np.linalg.norm(r_site)
    return np.arcsin(np.dot(rho_hat, zenith_hat))  # radians

def run_ekf(
    data: np.ndarray, apply_measurement_updates: bool = True
) -> Dict[str, np.ndarray]:
    """Execute the EKF across all measurements."""
    times = data[:, 0]
    stations = data[:, 1].astype(int)
    measurements = data[:, 2:4]
    x0, P0, accel_noise_std, R = part3_initial_conditions()
    x = x0.copy()
    P = P0.copy()
    x_minus = []
    P_minus = []
    x_plus = []
    P_plus = []
    innovations = []
    residuals = []
    yhat_minus = []
    yhat_plus = []
    t_prev = times[0]
    first_step = True
    for tk, station, meas in zip(times, stations, measurements):
        if first_step:
            dt = 0.0
            first_step = False
        else:
            dt = tk - t_prev
        if dt < 0:
            raise ValueError("Measurement times must be non-decreasing.")
        if dt > 0:
            x, phi = discrete_state_transition(t_prev, tk, x)
            Qk = process_noise_matrix(dt, accel_noise_std)
            P = phi @ P @ phi.T + Qk
        else:
            phi = np.eye(6)
        x_minus.append(x.copy())
        P_minus.append(P.copy())
        y_pred_minus = measurement_function(x, tk, station)
        H = measurement_jacobian(x, tk, station)
        S = H @ P @ H.T + R
        innovation = meas - y_pred_minus

        do_update = apply_measurement_updates

        # ---- NIS consistency check ----
        if do_update:
            nu = innovation
            nis = nu.T @ np.linalg.inv(S) @ nu
            if nis > 9.21:  # chi-square, 2 DOF, 99%
                do_update = False

        # ---- Measurement update ----
        if do_update:
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ innovation
            I = np.eye(6)
            P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
        else:
            K = np.zeros((6, 2))
        y_pred_plus = measurement_function(x, tk, station)
        residual = meas - y_pred_plus
        x_plus.append(x.copy())
        P_plus.append(P.copy())
        innovations.append(innovation)
        residuals.append(residual)
        yhat_minus.append(y_pred_minus)
        yhat_plus.append(y_pred_plus)
        t_prev = tk
    return {
        "times": times,
        "stations": stations,
        "x_minus": np.vstack(x_minus),
        "P_minus": np.stack(P_minus),
        "x_plus": np.vstack(x_plus),
        "P_plus": np.stack(P_plus),
        "innovations": np.vstack(innovations),
        "residuals": np.vstack(residuals),
        "yhat_minus": np.vstack(yhat_minus),
        "yhat_plus": np.vstack(yhat_plus),
    }


def part3_prediction_only(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the EKF prediction step only across all measurement times."""
    results = run_ekf(data, apply_measurement_updates=False)
    return results["times"], results["x_minus"], results["P_minus"]


def part3_plot_covariance(times: np.ndarray, covariances: np.ndarray) -> plt.Figure:
    """Plot ±3σ envelopes for each state component."""
    diag_entries = np.array([np.diag(P) for P in covariances])
    sigma = np.sqrt(diag_entries)
    labels = [
        "x (km)",
        "y (km)",
        "z (km)",
        "vx (km/s)",
        "vy (km/s)",
        "vz (km/s)",
    ]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for idx, ax in enumerate(axes.flatten()):
        upper = 3.0 * sigma[:, idx]
        lower = -upper
        ax.plot(times, upper, label="+3σ")
        ax.plot(times, lower, label="-3σ")
        ax.axhline(0.0, color="k", linewidth=0.8)
        ax.set_ylabel(labels[idx])
        max_span = np.max(np.abs(upper))
        if max_span > 0:
            ax.set_ylim(-1.1 * max_span, 1.1 * max_span)
        ax.grid(True)
        if idx == 0:
            ax.legend()
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle("Part 3: Prediction-only ±3σ Covariance Bounds")
    plt.tight_layout()
    return fig


def run_part3_prediction_plots(data: Optional[np.ndarray] = None) -> plt.Figure:
    """Convenience wrapper to run the Part 3 workflow."""
    if data is None:
        data = load_numpy_data("Project-Measurements-Easy.npy")
    times, _, P_history = part3_prediction_only(data)
    return part3_plot_covariance(times, P_history)


# ---- Part 4: Measurement updates ----
def part4_plot_pre_post_covariance(
    times: np.ndarray, P_minus: np.ndarray, P_plus: np.ndarray
) -> plt.Figure:
    """Plot pre- and post-update ±3σ bounds."""
    sigma_minus = np.sqrt(np.array([np.diag(P) for P in P_minus]))
    sigma_plus = np.sqrt(np.array([np.diag(P) for P in P_plus]))
    labels = ["x", "y", "z", "vx", "vy", "vz"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for idx, ax in enumerate(axes.flatten()):
        ax.plot(times, 3.0 * sigma_plus[:, idx],
                color="blue", linewidth=2.0, label="+3σ post" if idx == 0 else None)
        ax.plot(times, -3.0 * sigma_plus[:, idx],
                color="blue", linewidth=2.0, label="-3σ post" if idx == 0 else None)

        ax.plot(times, 3.0 * sigma_minus[:, idx],
                color="red", linestyle="--", linewidth=2.5, alpha=0.9,
                label="+3σ pre" if idx == 0 else None)
        ax.plot(times, -3.0 * sigma_minus[:, idx],
                color="red", linestyle="--", linewidth=2.5, alpha=0.9,
                label="-3σ pre" if idx == 0 else None)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Part 4b: Pre- vs Post-Update ±3σ Bounds")
    plt.tight_layout()
    return fig


def part4_plot_state_difference(
    times: np.ndarray, x_minus: np.ndarray, x_plus: np.ndarray, P_minus: np.ndarray
) -> plt.Figure:
    """Plot µ+ - µ- inside pre-update ±3σ bounds."""
    delta = x_plus - x_minus
    sigma_minus = np.sqrt(np.array([np.diag(P) for P in P_minus]))
    labels = ["x", "y", "z", "vx", "vy", "vz"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for idx, ax in enumerate(axes.flatten()):
        ax.plot(times, delta[:, idx], label="µ+ - µ-")
        bound = 3.0 * sigma_minus[:, idx]
        ax.plot(times, bound, "k--", label="+3σ pre" if idx == 0 else None)
        ax.plot(times, -bound, "k--", label="-3σ pre" if idx == 0 else None)
        ax.set_ylabel(labels[idx])
        ax.grid(True)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Part 4c: State Update Difference within Pre-Update Bounds")
    plt.tight_layout()
    return fig


# ---- Part 5: Filter solutions ----
def part5_plot_residuals(
    times: np.ndarray, stations: np.ndarray, residuals: np.ndarray
) -> plt.Figure:
    """Post-fit measurement residuals as a function of time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for station in np.unique(stations):
        mask = stations == station
        ax1.plot(times[mask], residuals[mask, 0], ".", label=f"DSN {station}")
        ax2.plot(times[mask], residuals[mask, 1], ".", label=f"DSN {station}")
    ax1.axhline(0.0, color="k", linewidth=0.8)
    ax2.axhline(0.0, color="k", linewidth=0.8)
    ax1.set_ylabel("Range Residual δρ (km)")
    ax2.set_ylabel("Range-rate Residual δρ̇ (km/s)")
    ax2.set_xlabel("Time (s)")
    ax1.grid(True)
    ax2.grid(True)
    ax1.legend()
    ax2.legend()
    fig.suptitle("Part 5a: Post-fit Measurement Residuals")
    plt.tight_layout()
    return fig

def part5_plot_residuals(
    times: np.ndarray, stations: np.ndarray, residuals: np.ndarray
) -> plt.Figure:
    """Post-fit measurement residuals as a function of time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for station in np.unique(stations):
        mask = stations == station
        ax1.plot(times[mask], residuals[mask, 0], ".", label=f"DSN {station}")
        ax2.plot(times[mask], residuals[mask, 1], ".", label=f"DSN {station}")
    ax1.axhline(0.0, color="k", linewidth=0.8)
    ax2.axhline(0.0, color="k", linewidth=0.8)
    ax1.set_ylabel("Range Residual δρ (km)")
    ax2.set_ylabel("Range-rate Residual δρ̇ (km/s)")
    ax2.set_xlabel("Time (s)")
    ax1.grid(True)
    ax2.grid(True)
    ax1.legend()
    ax2.legend()
    fig.suptitle("Part 5a: Post-fit Measurement Residuals")
    plt.tight_layout()
    return fig


def part5_plot_state_with_bounds(
    times: np.ndarray, x_plus: np.ndarray, P_plus: np.ndarray
) -> plt.Figure:
    """Estimated state with ±3σ bounds."""
    sigma = np.sqrt(np.array([np.diag(P) for P in P_plus]))
    labels = ["x (km)", "y (km)", "z (km)", "vx (km/s)", "vy (km/s)", "vz (km/s)"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for idx, ax in enumerate(axes.flatten()):
        ax.plot(times, x_plus[:, idx], label="μ+")
        bound = 3.0 * sigma[:, idx]
        ax.plot(times, x_plus[:, idx] + bound, "r--", label="+3σ" if idx == 0 else None)
        ax.plot(times, x_plus[:, idx] - bound, "r--", label="-3σ" if idx == 0 else None)
        ax.set_ylabel(labels[idx])
        ax.grid(True)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Part 5c: Estimated State with ±3σ Bounds")
    plt.tight_layout()
    return fig


def part5_report_final_state(times: np.ndarray, x_plus: np.ndarray, P_plus: np.ndarray) -> None:
    """Print final state estimate and uncertainty."""
    final_time = times[-1]
    final_state = x_plus[-1]
    final_sigma = np.sqrt(np.diag(P_plus[-1]))
    components = ["x", "y", "z", "vx", "vy", "vz"]
    print("\nPart 5d: Final state estimate at t = {:.1f} s".format(final_time))
    for name, value, sigma in zip(components, final_state, final_sigma):
        print(f"  {name}: {value:.6f} ± {3*sigma:.6f} ({sigma:.6f} 1σ)")


def evaluate_residuals(residuals: np.ndarray) -> None:
    """Provide a quick assessment of residual magnitudes vs sensor noise."""
    sigma_r = 1e-3
    sigma_rdot = 1e-5
    rms_range = np.sqrt(np.mean(residuals[:, 0] ** 2))
    rms_rdot = np.sqrt(np.mean(residuals[:, 1] ** 2))
    print("\nPart 5b: Residual assessment")
    print(f"  Range RMS residual: {rms_range:.6e} km (sensor σ ≈ {sigma_r:.6e} km)")
    print(f"  Range-rate RMS residual: {rms_rdot:.6e} km/s (sensor σ ≈ {sigma_rdot:.6e} km/s)")

def main():
    print_part2_pseudocode()
    data = load_numpy_data("Project-Measurements-Easy.npy")
    part1e_plot_measurements(data)
    run_part3_prediction_plots(data)
    ekf_results = run_ekf(data, apply_measurement_updates=True)
    part4_plot_pre_post_covariance(
        ekf_results["times"], ekf_results["P_minus"], ekf_results["P_plus"]
    )
    part4_plot_state_difference(
        ekf_results["times"], ekf_results["x_minus"], ekf_results["x_plus"], ekf_results["P_minus"]
    )
    part5_plot_residuals(ekf_results["times"], ekf_results["stations"], ekf_results["residuals"])
    part5_plot_state_with_bounds(ekf_results["times"], ekf_results["x_plus"], ekf_results["P_plus"])
    evaluate_residuals(ekf_results["residuals"])
    part5_report_final_state(ekf_results["times"], ekf_results["x_plus"], ekf_results["P_plus"])
    plt.show()



if __name__ == "__main__":
    main()
