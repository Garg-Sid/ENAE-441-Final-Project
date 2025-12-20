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
def part3_initial_conditions(
    pos_sigma_km: float = 10.0,
    vel_sigma_kms: float = 0.05,
    accel_noise_std: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
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
    P0 = np.diag(
        [
            pos_sigma_km**2,
            pos_sigma_km**2,
            pos_sigma_km**2,
            vel_sigma_kms**2,
            vel_sigma_kms**2,
            vel_sigma_kms**2,
        ]
    )
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
    data: np.ndarray,
    apply_measurement_updates: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Execute the EKF across all measurements.
    Innovations use x⁻, residuals use x⁺.
    Saves S_k for NIS testing.
    """

    times        = data[:, 0]
    stations     = data[:, 1].astype(int)
    measurements = data[:, 2:4]

    x0, P0, accel_sigma, R = part3_initial_conditions()

    N = len(times)
    n = len(x0)

    # Storage
    x_minus = np.zeros((N, n))
    P_minus = np.zeros((N, n, n))
    x_plus  = np.zeros((N, n))
    P_plus  = np.zeros((N, n, n))

    innovations = np.zeros((N, 2))   # y − h(x−)
    residuals   = np.zeros((N, 2))   # y − h(x+)
    S_all       = np.zeros((N, 2, 2))

    # Initial
    x = x0.copy()
    P = P0.copy()
    t_prev = times[0]

    I = np.eye(n)

    for k, (tk, station, meas) in enumerate(zip(times, stations, measurements)):

        # Prediction step
        # 
        dt = tk - t_prev if k > 0 else 0.0

        if dt > 0.0:
            x, Phi = discrete_state_transition(t_prev, tk, x)
            Qk = process_noise_matrix(dt, accel_sigma)
            P = Phi @ P @ Phi.T + Qk

        x_minus[k, :] = x
        P_minus[k, :, :] = P

        # Measurement prediction
        # 
        y_hat_minus = measurement_function(x, tk, station)
        Hk = measurement_jacobian(x, tk, station)

        Sk = Hk @ P @ Hk.T + R
        S_all[k, :, :] = Sk

        nu = meas - y_hat_minus
        innovations[k, :] = nu

        # Measurement update
        # 
        if apply_measurement_updates:
            # Kalman gain (no explicit inverse)
            Kk = P @ Hk.T @ np.linalg.solve(Sk, np.eye(2))

            x = x + Kk @ nu

            # Joseph stabilized covariance update
            P = (I - Kk @ Hk) @ P @ (I - Kk @ Hk).T + Kk @ R @ Kk.T

        x_plus[k, :] = x
        P_plus[k, :, :] = P

        # Post-fit residuals (CRITICAL)
        # 
        y_hat_plus = measurement_function(x, tk, station)
        residuals[k, :] = meas - y_hat_plus

        t_prev = tk

    return {
        "times": times,
        "stations": stations,

        "x_minus": x_minus,
        "P_minus": P_minus,
        "x_plus":  x_plus,
        "P_plus":  P_plus,

        "innovations": innovations,
        "residuals": residuals,
        "S": S_all,
    }





def part3_prediction_only(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the EKF prediction step only across all measurement times."""
    results = run_ekf(data, apply_measurement_updates=False)
    return results["times"], results["x_minus"], results["P_minus"]


def part3_plot_covariance(
    times: np.ndarray, states: np.ndarray, covariances: np.ndarray
) -> plt.Figure:
    """Plot predicted state with ±3σ envelopes for each component."""
    diag_entries = np.array([np.diag(P) for P in covariances])
    sigma = np.sqrt(diag_entries)
    pos_labels = ["X Pos (km)", "Y Pos (km)", "Z Pos (km)"]
    vel_labels = ["X Vel (km/s)", "Y Vel (km/s)", "Z Vel (km/s)"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True)
    for col in range(3):
        idx_pos = col
        idx_vel = col + 3

        ax_pos = axes[0, col]
        mu_pos = states[:, idx_pos]
        bound_pos = 3.0 * sigma[:, idx_pos]
        ax_pos.plot(times, mu_pos, color="tab:blue", linewidth=1.5, label="State")
        ax_pos.fill_between(
            times,
            mu_pos - bound_pos,
            mu_pos + bound_pos,
            color="tab:blue",
            alpha=0.2,
            label="±3σ",
        )
        ax_pos.set_ylabel(pos_labels[col])
        ax_pos.grid(True)
        ax_pos.legend(loc="upper left")

        ax_vel = axes[1, col]
        mu_vel = states[:, idx_vel]
        bound_vel = 3.0 * sigma[:, idx_vel]
        ax_vel.plot(times, mu_vel, color="tab:blue", linewidth=1.5, label="State")
        ax_vel.fill_between(
            times,
            mu_vel - bound_vel,
            mu_vel + bound_vel,
            color="tab:blue",
            alpha=0.2,
            label="±3σ",
        )
        ax_vel.set_ylabel(vel_labels[col])
        ax_vel.set_xlabel("Time (s)")
        ax_vel.grid(True)
        ax_vel.legend(loc="upper left")
    fig.suptitle("Part 3c: Pure Prediction of State Over Time")
    plt.tight_layout()
    return fig


def run_part3_prediction_plots(data: Optional[np.ndarray] = None) -> plt.Figure:
    """Convenience wrapper to run the Part 3 workflow."""
    if data is None:
        data = load_numpy_data("Project-Measurements-Easy.npy")
    times, x_history, P_history = part3_prediction_only(data)
    return part3_plot_covariance(times, x_history, P_history)


def part4_plot_pre_post_covariance(
    times: np.ndarray, P_minus: np.ndarray, P_plus: np.ndarray
) -> plt.Figure:
    """Part 4b: Pre- and post-update ±3σ bounds with clear styling separation."""
    sigma_minus = np.sqrt(np.array([np.diag(P) for P in P_minus]))
    sigma_plus  = np.sqrt(np.array([np.diag(P) for P in P_plus]))
    pos_labels = ["X Pos (km)", "Y Pos (km)", "Z Pos (km)"]
    vel_labels = ["X Vel (km/s)", "Y Vel (km/s)", "Z Vel (km/s)"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True)

    for col in range(3):
        idx_pos = col
        idx_vel = col + 3

        ax_pos = axes[0, col]
        pre_pos = 3.0 * sigma_minus[:, idx_pos]
        post_pos = 3.0 * sigma_plus[:, idx_pos]

        ax_pos.plot(times,  post_pos, linewidth=1.8, label="+3σ post" if col == 0 else None)
        ax_pos.plot(times, -post_pos, linewidth=1.8, label="-3σ post" if col == 0 else None)
        ax_pos.plot(times,  pre_pos, linestyle="--", linewidth=1.4, alpha=0.9,
                    label="+3σ pre" if col == 0 else None)
        ax_pos.plot(times, -pre_pos, linestyle="--", linewidth=1.4, alpha=0.9,
                    label="-3σ pre" if col == 0 else None)

        ax_pos.set_ylabel(pos_labels[col])
        ax_pos.grid(True)

        ax_vel = axes[1, col]
        pre_vel = 3.0 * sigma_minus[:, idx_vel]
        post_vel = 3.0 * sigma_plus[:, idx_vel]

        ax_vel.plot(times,  post_vel, linewidth=1.8, label="+3σ post" if col == 0 else None)
        ax_vel.plot(times, -post_vel, linewidth=1.8, label="-3σ post" if col == 0 else None)
        ax_vel.plot(times,  pre_vel, linestyle="--", linewidth=1.4, alpha=0.9,
                    label="+3σ pre" if col == 0 else None)
        ax_vel.plot(times, -pre_vel, linestyle="--", linewidth=1.4, alpha=0.9,
                    label="-3σ pre" if col == 0 else None)

        ax_vel.set_ylabel(vel_labels[col])
        ax_vel.set_xlabel("Time (s)")
        ax_vel.grid(True)

    axes[0, 0].legend(loc="upper right")

    fig.suptitle("Part 4b: Pre- vs Post-Update ±3σ Bounds")
    plt.tight_layout()
    return fig

def part4_plot_state_difference(
    times: np.ndarray, x_minus: np.ndarray, x_plus: np.ndarray, P_minus: np.ndarray
) -> plt.Figure:
    """Part 4c: (μ+ − μ−) overlaid with pre-update ±3σ bounds."""
    delta = x_plus - x_minus
    sigma_minus = np.sqrt(np.array([np.diag(P) for P in P_minus]))
    pos_labels = ["X Pos (km)", "Y Pos (km)", "Z Pos (km)"]
    vel_labels = ["X Vel (km/s)", "Y Vel (km/s)", "Z Vel (km/s)"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True)

    for col in range(3):
        idx_pos = col
        idx_vel = col + 3

        ax_pos = axes[0, col]
        bound_pos = 3.0 * sigma_minus[:, idx_pos]

        ax_pos.plot(times, delta[:, idx_pos], linewidth=1.6, label="μ+ − μ−" if col == 0 else None)
        ax_pos.plot(times,  bound_pos, linestyle="--", linewidth=1.2, alpha=0.9,
                    label="+3σ pre" if col == 0 else None)
        ax_pos.plot(times, -bound_pos, linestyle="--", linewidth=1.2, alpha=0.9,
                    label="-3σ pre" if col == 0 else None)

        ax_pos.set_ylabel(pos_labels[col])
        ax_pos.grid(True)

        ax_vel = axes[1, col]
        bound_vel = 3.0 * sigma_minus[:, idx_vel]

        ax_vel.plot(times, delta[:, idx_vel], linewidth=1.6, label="μ+ − μ−" if col == 0 else None)
        ax_vel.plot(times,  bound_vel, linestyle="--", linewidth=1.2, alpha=0.9,
                    label="+3σ pre" if col == 0 else None)
        ax_vel.plot(times, -bound_vel, linestyle="--", linewidth=1.2, alpha=0.9,
                    label="-3σ pre" if col == 0 else None)

        ax_vel.set_ylabel(vel_labels[col])
        ax_vel.set_xlabel("Time (s)")
        ax_vel.grid(True)

    axes[0, 0].legend(loc="upper right")

    fig.suptitle("Part 4c: State Update Difference within Pre-Update Bounds")
    plt.tight_layout()
    return fig



# ---- Part 5: Filter solutions ----
def part5_plot_residuals(
    times: np.ndarray, stations: np.ndarray, residuals: np.ndarray
) -> plt.Figure:
    """
    Part 5a: Post-fit measurement residuals vs time.
    Plotted in meters and cm/s for readability, with scatter (no connecting lines).
    """
    # Convert to intuitive units
    dr_m = residuals[:, 0] * 1e3          # km -> m
    drdot_cms = residuals[:, 1] * 1e5     # km/s -> cm/s

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for station in np.unique(stations):
        mask = stations == station

        ax1.scatter(times[mask], dr_m[mask], s=10, label=f"DSN {station}", alpha=0.8)
        ax2.scatter(times[mask], drdot_cms[mask], s=10, label=f"DSN {station}", alpha=0.8)

    ax1.axhline(0.0, linewidth=0.8)
    ax2.axhline(0.0, linewidth=0.8)

    ax1.set_ylabel("Range Residual δρ (m)")
    ax2.set_ylabel("Range-rate Residual δρ̇ (cm/s)")
    ax2.set_xlabel("Time (s)")

    ax1.grid(True)
    ax2.grid(True)

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")

    fig.suptitle("Part 5a: Post-fit Measurement Residuals")
    plt.tight_layout()
    return fig


def part5_plot_state_with_bounds(
    times: np.ndarray, x_plus: np.ndarray, P_plus: np.ndarray
) -> plt.Figure:
    """Part 5c: Estimated state with ±3σ bounds (clear color/linestyle separation)."""
    sigma = np.sqrt(np.array([np.diag(P) for P in P_plus]))
    pos_labels = ["X Pos (km)", "Y Pos (km)", "Z Pos (km)"]
    vel_labels = ["X Vel (km/s)", "Y Vel (km/s)", "Z Vel (km/s)"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True)

    for col in range(3):
        idx_pos = col
        idx_vel = col + 3

        ax_pos = axes[0, col]
        mu_pos = x_plus[:, idx_pos]
        bound_pos = 3.0 * sigma[:, idx_pos]

        ax_pos.plot(times, mu_pos, linewidth=1.8, label="μ+" if col == 0 else None)
        ax_pos.plot(times, mu_pos + bound_pos, linestyle="--", linewidth=1.2, alpha=0.9,
                    label="+3σ" if col == 0 else None)
        ax_pos.plot(times, mu_pos - bound_pos, linestyle="--", linewidth=1.2, alpha=0.9,
                    label="-3σ" if col == 0 else None)

        ax_pos.set_ylabel(pos_labels[col])
        ax_pos.grid(True)

        ax_vel = axes[1, col]
        mu_vel = x_plus[:, idx_vel]
        bound_vel = 3.0 * sigma[:, idx_vel]

        ax_vel.plot(times, mu_vel, linewidth=1.8, label="μ+" if col == 0 else None)
        ax_vel.plot(times, mu_vel + bound_vel, linestyle="--", linewidth=1.2, alpha=0.9,
                    label="+3σ" if col == 0 else None)
        ax_vel.plot(times, mu_vel - bound_vel, linestyle="--", linewidth=1.2, alpha=0.9,
                    label="-3σ" if col == 0 else None)

        ax_vel.set_ylabel(vel_labels[col])
        ax_vel.set_xlabel("Time (s)")
        ax_vel.grid(True)

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


def ekf_performance_tests(results: Dict[str, np.ndarray], R: np.ndarray) -> None:
    times = results["times"]
    residuals = results["residuals"]          # y_k - yhat_plus
    innovations = results["innovations"]      # y_k - yhat_minus
    P_minus = results["P_minus"]
    x_minus = results["x_minus"]
    x_plus = results["x_plus"]
    S_all = results.get("S", None)

    print("\nEKF Performance Tests")

    # 1) Residual mean near zero
    r_mean = np.mean(residuals, axis=0)
    print("Residual mean [range, range-rate] (km, km/s):", r_mean)

    # 2) Residual covariance vs R (rough check)
    r_cov = np.cov(residuals.T, bias=False)
    print("\nResidual sample covariance:")
    print(r_cov)
    print("\nR used in filter:")
    print(R)

    # 3) Residual no structure check (lag-1 autocorrelation)
    def lag1_autocorr(x: np.ndarray) -> float:
        x0 = x[:-1] - np.mean(x[:-1])
        x1 = x[1:] - np.mean(x[1:])
        denom = np.sqrt(np.sum(x0**2) * np.sum(x1**2))
        return 0.0 if denom == 0 else float(np.sum(x0 * x1) / denom)

    ac_r = lag1_autocorr(residuals[:, 0])
    ac_rr = lag1_autocorr(residuals[:, 1])
    print("\nResidual lag-1 autocorr (range, range-rate):", ac_r, ac_rr)

    # 4) Proxy state error test: delta_x within 3sigma(pre)
    delta_x = x_plus - x_minus
    sigma_minus = np.sqrt(np.array([np.diag(P) for P in P_minus]))
    inside = np.abs(delta_x) <= 3.0 * sigma_minus
    frac_inside = np.mean(inside, axis=0)
    labels = ["x", "y", "z", "vx", "vy", "vz"]
    print("\nProxy state error fraction inside 3sigma(pre) by component:")
    for lab, frac in zip(labels, frac_inside):
        print(f"  {lab}: {frac:.4f}")

    # 5) NIS test (requires S_k saved)
    if S_all is None:
        print("\nNIS test skipped because results['S'] is not saved.")
        return

    # NIS_k = nu_k^T S_k^{-1} nu_k
    nis = np.zeros(len(times))
    for k in range(len(times)):
        nu = innovations[k].reshape(2, 1)
        S = S_all[k]
        nis[k] = (nu.T @ np.linalg.solve(S, nu)).item()

    nis_mean = np.mean(nis)
    print("\nNIS mean:", nis_mean, "Expected near p=2")

    # Optional chi-square consistency window (95% and 99.7% style)
    from scipy.stats import chi2
    p = 2
    lo95, hi95 = chi2.ppf(0.025, p), chi2.ppf(0.975, p)
    lo997, hi997 = chi2.ppf(0.0015, p), chi2.ppf(0.9985, p)

    frac95 = np.mean((nis >= lo95) & (nis <= hi95))
    frac997 = np.mean((nis >= lo997) & (nis <= hi997))

    print("NIS 95% window bounds:", lo95, hi95, "fraction inside:", frac95)
    print("NIS 99.7% window bounds:", lo997, hi997, "fraction inside:", frac997)

def propagate_dense_prediction(
    times: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
    accel_noise_std: float,
    n_substeps: int = 50,
):
    """Dense prediction-only propagation for smooth state and covariance plots."""

    t_dense = [times[0]]
    x_dense = [x0.copy()]
    P_dense = [P0]

    x = x0.copy()
    P = P0.copy()
    t_prev = times[0]

    for tk in times[1:]:
        dt = tk - t_prev
        if dt <= 0:
            continue

        dt_sub = dt / n_substeps

        for i in range(n_substeps):
            t0 = t_prev + i * dt_sub
            t1 = t0 + dt_sub

            x, phi = discrete_state_transition(t0, t1, x)
            Qk = process_noise_matrix(dt_sub, accel_noise_std)
            P = phi @ P @ phi.T + Qk

            t_dense.append(t1)
            x_dense.append(x.copy())
            P_dense.append(P)

        t_prev = tk

    return np.array(t_dense), np.array(x_dense), np.array(P_dense)

def part5c_plot_prediction_bounds_smooth(
    times: np.ndarray, x0: np.ndarray, P0: np.ndarray, accel_noise_std: float
) -> plt.Figure:
    """
    Prediction-only state with ±3σ covariance bounds using dense propagation.
    Blue solid lines = state, shaded band = ±3σ.
    """

    t_dense, x_dense, P_dense = propagate_dense_prediction(
        times, x0, P0, accel_noise_std, n_substeps=50
    )

    sigma = np.sqrt(np.array([np.diag(P) for P in P_dense]))
    pos_labels = ["X Pos (km)", "Y Pos (km)", "Z Pos (km)"]
    vel_labels = ["X Vel (km/s)", "Y Vel (km/s)", "Z Vel (km/s)"]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True)

    for col in range(3):
        idx_pos = col
        idx_vel = col + 3

        ax_pos = axes[0, col]
        mu_pos = x_dense[:, idx_pos]
        bound_pos = 3.0 * sigma[:, idx_pos]
        ax_pos.plot(t_dense, mu_pos, color="tab:blue", linewidth=1.5, label="State")
        ax_pos.fill_between(
            t_dense,
            mu_pos - bound_pos,
            mu_pos + bound_pos,
            color="tab:blue",
            alpha=0.2,
            label="±3σ",
        )
        ax_pos.set_ylabel(pos_labels[col])
        ax_pos.grid(True)
        ax_pos.legend(loc="upper left")

        ax_vel = axes[1, col]
        mu_vel = x_dense[:, idx_vel]
        bound_vel = 3.0 * sigma[:, idx_vel]
        ax_vel.plot(t_dense, mu_vel, color="tab:blue", linewidth=1.5, label="State")
        ax_vel.fill_between(
            t_dense,
            mu_vel - bound_vel,
            mu_vel + bound_vel,
            color="tab:blue",
            alpha=0.2,
            label="±3σ",
        )
        ax_vel.set_ylabel(vel_labels[col])
        ax_vel.set_xlabel("Time (s)")
        ax_vel.grid(True)
        ax_vel.legend(loc="upper left")

    fig.suptitle("Part 3c: Pure Prediction of State Over Time")
    plt.tight_layout()
    return fig


def main():


    # Load data
    # 
    # print_part2_pseudocode()
    data = load_numpy_data("Project-Measurements-Easy.npy")


    # Part 1e: Raw measurements
    # 
    part1e_plot_measurements(data)

    # Part 3: Prediction-only (dense propagation)
    #
    x0, P0, accel_sigma, _ = part3_initial_conditions()

    fig = part5c_plot_prediction_bounds_smooth(
        data[:, 0], x0, P0, accel_sigma
    )
    fig.suptitle("Part 3c: Pure Prediction of State Over Time")

    # Run EKF with measurement updates
    # 
    ekf_results = run_ekf(data, apply_measurement_updates=True)

    # Part 4b: Pre- vs post-update covariance
    # 
    part4_plot_pre_post_covariance(
        ekf_results["times"],
        ekf_results["P_minus"],
        ekf_results["P_plus"],
    )

    # Part 4c: State update difference
    # 
    part4_plot_state_difference(
        ekf_results["times"],
        ekf_results["x_minus"],
        ekf_results["x_plus"],
        ekf_results["P_minus"],
    )

    # Part 5a: Post-fit residuals
    # 
    part5_plot_residuals(
        ekf_results["times"],
        ekf_results["stations"],
        ekf_results["residuals"],
    )

    # Part 5c: Estimated state with bounds
    #
    part5_plot_state_with_bounds(
        ekf_results["times"],
        ekf_results["x_plus"],
        ekf_results["P_plus"],
    )


    # Part 5d: Final state estimate
    # 
    part5_report_final_state(
        ekf_results["times"],
        ekf_results["x_plus"],
        ekf_results["P_plus"],
    )

    # Performance tests (NIS, RMS, etc.)
    #
    # ekf_performance_tests(
    #    ekf_results,
    #    measurement_noise_matrix()
    # )

    # Show all figures
    # 
    plt.show()


if __name__ == "__main__":
    main()
