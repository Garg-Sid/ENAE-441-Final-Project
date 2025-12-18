"""
ENGR 441 Final Project – EKF for 2-body orbit determination with DSN range / range-rate.

This script is a cleaned + debugged version of your Main.py:
- keeps your modeling choices (2-body, spherical rotating Earth, DSN sites, range & range-rate)
- adds: better numerics, clearer structure, optional outlier gating, easy/hard switch, and reporting helpers
- fixes/clarifies: Part 4c bound choice (can plot inside P_minus OR P_plus, selectable)

Units:
- position km, velocity km/s, time s, angles rad
- measurement noise: 1 m -> 1e-3 km, 1 cm/s -> 1e-5 km/s
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -----------------------
# Constants (from prompt)
# -----------------------
MU_EARTH = 398600.4418          # km^3/s^2
R_EARTH = 6378.137              # km
OMEGA_E = 7.292115e-5           # rad/s
GAMMA0 = 0.0                    # rad (Local Sidereal Time at t0)
DEG2RAD = np.pi / 180.0

DSN_SITES = {
    0: {"name": "Goldstone", "lat": 35.297 * DEG2RAD,  "lon": -116.914 * DEG2RAD},
    1: {"name": "Madrid",    "lat": 40.4311 * DEG2RAD, "lon": -4.248 * DEG2RAD},
    2: {"name": "Canberra",  "lat": -35.4023 * DEG2RAD,"lon": 148.9813 * DEG2RAD},
}

# -----------------------
# Data I/O
# -----------------------
def load_measurements(npy_name: str) -> np.ndarray:
    """Load [t, station_i, rho(km), rhodot(km/s)] rows."""
    cur_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    arr = np.load(cur_dir + npy_name, allow_pickle=True)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"Expected Nx4 array, got {arr.shape}")
    # ensure time-sorted
    arr = arr[np.argsort(arr[:, 0])]
    return arr

# -----------------------
# Dynamics & measurements
# -----------------------
def two_body_dynamics(_t: float, x: np.ndarray) -> np.ndarray:
    r = x[:3]
    v = x[3:]
    rn = np.linalg.norm(r)
    if rn <= 0:
        raise ValueError("||r|| <= 0 in dynamics")
    a = -MU_EARTH * r / rn**3
    return np.hstack((v, a))

def A_jacobian(x: np.ndarray) -> np.ndarray:
    r = x[:3]
    rn = np.linalg.norm(r)
    if rn <= 0:
        raise ValueError("||r|| <= 0 in A_jacobian")
    I3 = np.eye(3)
    rrT = np.outer(r, r)
    dadr = -MU_EARTH * (I3 / rn**3 - 3.0 * rrT / rn**5)
    top = np.hstack((np.zeros((3, 3)), I3))
    bot = np.hstack((dadr, np.zeros((3, 3))))
    return np.vstack((top, bot))

def site_inertial(station: int, t: float) -> Tuple[np.ndarray, np.ndarray]:
    """Station inertial position and velocity under simple Earth rotation about +z."""
    s = DSN_SITES[int(station)]
    lat, lon = s["lat"], s["lon"]
    theta = lon + OMEGA_E * t + GAMMA0
    cphi = np.cos(lat)
    r_site = R_EARTH * np.array([cphi*np.cos(theta), cphi*np.sin(theta), np.sin(lat)])
    omega = np.array([0.0, 0.0, OMEGA_E])
    v_site = np.cross(omega, r_site)
    return r_site, v_site

def h_measure(x: np.ndarray, t: float, station: int) -> np.ndarray:
    r, v = x[:3], x[3:]
    r_site, v_site = site_inertial(station, t)
    rho_vec = r - r_site
    rho = np.linalg.norm(rho_vec)
    if rho <= 0:
        raise ValueError("rho <= 0 in measurement")
    rhodot = rho_vec @ (v - v_site) / rho
    return np.array([rho, rhodot])

def H_jacobian(x: np.ndarray, t: float, station: int) -> np.ndarray:
    r, v = x[:3], x[3:]
    r_site, v_site = site_inertial(station, t)
    rho_vec = r - r_site
    rho = np.linalg.norm(rho_vec)
    if rho <= 0:
        raise ValueError("rho <= 0 in H_jacobian")

    drho_dr = rho_vec / rho
    v_rel = v - v_site
    rhodot = rho_vec @ v_rel / rho

    # d(rhodot)/dr and d(rhodot)/dv
    drhodot_dr = v_rel / rho - (rhodot / rho**2) * rho_vec
    drhodot_dv = rho_vec / rho

    H = np.zeros((2, 6))
    H[0, :3] = drho_dr
    H[1, :3] = drhodot_dr
    H[1, 3:] = drhodot_dv
    return H

# -----------------------
# Discretization (Φ)
# -----------------------
def propagate_with_phi(t0: float, t1: float, x0: np.ndarray, rtol=1e-9, atol=1e-9) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate state and variational equation to get x(t1), Φ(t1,t0)."""
    def ode(t, y):
        x = y[:6]
        phi = y[6:].reshape(6, 6)
        xdot = two_body_dynamics(t, x)
        phidot = A_jacobian(x) @ phi
        return np.hstack((xdot, phidot.ravel()))

    y0 = np.hstack((x0, np.eye(6).ravel()))
    sol = solve_ivp(ode, (t0, t1), y0, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError("propagate_with_phi failed")
    x1 = sol.y[:6, -1]
    phi = sol.y[6:, -1].reshape(6, 6)
    return x1, phi

# -----------------------
# Noise models
# -----------------------
def R_meas() -> np.ndarray:
    """From prompt: σρ^2 = 1 m^2, σρdot^2 = (1 cm/s)^2."""
    sigma_r_km = 1e-3
    sigma_rdot_kms = 1e-5
    return np.diag([sigma_r_km**2, sigma_rdot_kms**2])

def Q_process(dt: float, accel_sigma_kms2: float) -> np.ndarray:
    """
    Discrete Q from white acceleration noise in each axis.
    x=[r;v]. Continuous accel noise variance = accel_sigma^2.
    """
    q = accel_sigma_kms2**2
    I3 = np.eye(3)
    Qrr = (dt**3 / 3.0) * I3
    Qrv = (dt**2 / 2.0) * I3
    Qvv = dt * I3
    top = np.hstack((Qrr, Qrv))
    bot = np.hstack((Qrv, Qvv))
    return q * np.vstack((top, bot))

# -----------------------
# Initial condition helper
# -----------------------
def rot1(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rot3(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def coe_to_rv(a_km: float, e: float, inc: float, argp: float, raan: float, nu: float) -> Tuple[np.ndarray, np.ndarray]:
    p = a_km * (1 - e**2)
    r_pf = (p / (1 + e*np.cos(nu))) * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = np.sqrt(MU_EARTH / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    C = rot3(raan) @ rot1(inc) @ rot3(argp)
    return C @ r_pf, C @ v_pf

# -----------------------
# EKF
# -----------------------
@dataclass
class EKFConfig:
    accel_sigma_kms2: float = 1e-5      # tuning (unmodeled accel)
    min_elev_deg: Optional[float] = 5.0 # elevation gate (None disables)
    nis_gate: Optional[float] = None    # e.g. 9.21 for chi2(2) 99%; None disables
    joseph_form: bool = True

def elevation_rad(x: np.ndarray, t: float, station: int) -> float:
    r = x[:3]
    r_site, _ = site_inertial(station, t)
    rho = r - r_site
    rho_hat = rho / np.linalg.norm(rho)
    zenith = r_site / np.linalg.norm(r_site)
    return np.arcsin(rho_hat @ zenith)

def run_ekf(meas: np.ndarray,
            x0: np.ndarray,
            P0: np.ndarray,
            cfg: EKFConfig) -> dict:
    t = meas[:, 0]
    st = meas[:, 1].astype(int)
    y = meas[:, 2:4]
    R = R_meas()

    x = x0.copy()
    P = P0.copy()
    tprev = t[0]

    x_minus, P_minus = [], []
    x_plus,  P_plus  = [], []
    innovs, resids   = [], []
    upd_mask         = []

    for tk, sk, yk in zip(t, st, y):
        dt = float(tk - tprev)
        if dt < 0:
            raise ValueError("Measurement times must be nondecreasing")

        # Predict
        if dt > 0:
            x, Phi = propagate_with_phi(tprev, tk, x)
            P = Phi @ P @ Phi.T + Q_process(dt, cfg.accel_sigma_kms2)
        else:
            Phi = np.eye(6)

        x_minus.append(x.copy())
        P_minus.append(P.copy())

        # Measurement prediction
        yhat = h_measure(x, tk, sk)
        H = H_jacobian(x, tk, sk)
        nu = yk - yhat
        S = H @ P @ H.T + R

        # Decide update
        do_update = True
        if cfg.min_elev_deg is not None:
            do_update = do_update and (elevation_rad(x, tk, sk) > cfg.min_elev_deg * DEG2RAD)

        if do_update and cfg.nis_gate is not None:
            # NIS = nu^T S^{-1} nu
            nis = float(nu.T @ np.linalg.solve(S, nu))
            do_update = do_update and (nis <= cfg.nis_gate)

        if do_update:
            # K = P H^T S^{-1}
            K = (np.linalg.solve(S, (P @ H.T).T)).T
            x = x + K @ nu
            if cfg.joseph_form:
                I = np.eye(6)
                P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
            else:
                P = (np.eye(6) - K @ H) @ P
        else:
            K = np.zeros((6, 2))

        upd_mask.append(do_update)

        # Post-fit residual (Part 5a)
        yhat_plus = h_measure(x, tk, sk)
        resid = yk - yhat_plus

        x_plus.append(x.copy())
        P_plus.append(P.copy())
        innovs.append(nu)
        resids.append(resid)

        tprev = tk

    return {
        "t": t,
        "station": st,
        "x_minus": np.vstack(x_minus),
        "P_minus": np.stack(P_minus),
        "x_plus":  np.vstack(x_plus),
        "P_plus":  np.stack(P_plus),
        "innov":   np.vstack(innovs),
        "resid":   np.vstack(resids),
        "updated": np.array(upd_mask, dtype=bool),
    }

# -----------------------
# Plotting helpers (Parts 3–5)
# -----------------------
def plot_measurements(meas: np.ndarray) -> plt.Figure:
    t = meas[:,0]
    st = meas[:,1].astype(int)
    rho = meas[:,2]
    rhod = meas[:,3]
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,8), sharex=True)
    for s in np.unique(st):
        m = st==s
        ax1.plot(t[m], rho[m], ".", label=f"DSN {s}")
        ax2.plot(t[m], rhod[m], ".", label=f"DSN {s}")
    ax1.set_ylabel("Range ρ (km)")
    ax2.set_ylabel("Range-rate ρ̇ (km/s)")
    ax2.set_xlabel("Time (s)")
    ax1.grid(True); ax2.grid(True)
    ax1.legend(); ax2.legend()
    fig.suptitle("Range / Range-rate Measurements vs Time")
    fig.tight_layout()
    return fig

def plot_sigma_envelopes(t: np.ndarray, P: np.ndarray, title: str) -> plt.Figure:
    sig = np.sqrt(np.stack([np.diag(Pk) for Pk in P]))
    labels = ["x (km)","y (km)","z (km)","vx (km/s)","vy (km/s)","vz (km/s)"]
    fig, ax = plt.subplots(3,2,figsize=(12,10), sharex=True)
    for i, axi in enumerate(ax.ravel()):
        up = 3*sig[:,i]
        axi.plot(t, up, label="+3σ" if i==0 else None)
        axi.plot(t,-up, label="-3σ" if i==0 else None)
        axi.axhline(0, linewidth=0.8)
        axi.set_ylabel(labels[i])
        axi.grid(True)
    ax[-1,0].set_xlabel("Time (s)")
    ax[-1,1].set_xlabel("Time (s)")
    ax[0,0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig

def plot_pre_post_sigma(t: np.ndarray, Pm: np.ndarray, Pp: np.ndarray) -> plt.Figure:
    sm = np.sqrt(np.stack([np.diag(Pk) for Pk in Pm]))
    sp = np.sqrt(np.stack([np.diag(Pk) for Pk in Pp]))
    labels = ["x","y","z","vx","vy","vz"]
    fig, ax = plt.subplots(3,2,figsize=(12,10), sharex=True)
    for i, axi in enumerate(ax.ravel()):
        axi.plot(t, 3*sp[:,i], label="+3σ post" if i==0 else None)
        axi.plot(t,-3*sp[:,i], label="-3σ post" if i==0 else None)
        axi.plot(t, 3*sm[:,i], "--", label="+3σ pre" if i==0 else None)
        axi.plot(t,-3*sm[:,i], "--", label="-3σ pre" if i==0 else None)
        axi.set_ylabel(labels[i]); axi.grid(True)
    ax[-1,0].set_xlabel("Time (s)"); ax[-1,1].set_xlabel("Time (s)")
    ax[0,0].legend()
    fig.suptitle("Part 4b: Pre- vs Post-Update ±3σ Bounds")
    fig.tight_layout()
    return fig

def plot_state_update_vs_bounds(t: np.ndarray,
                               x_minus: np.ndarray,
                               x_plus: np.ndarray,
                               P_bounds: np.ndarray,
                               bounds_label: Literal["pre","post"]="pre") -> plt.Figure:
    """
    Part 4c:
      delta = μ+ - μ-
      Plot inside ±3σ computed from P_bounds (choose P_minus or P_plus).
    """
    delta = x_plus - x_minus
    sig = np.sqrt(np.stack([np.diag(Pk) for Pk in P_bounds]))
    labels = ["x","y","z","vx","vy","vz"]
    fig, ax = plt.subplots(3,2,figsize=(12,10), sharex=True)
    for i, axi in enumerate(ax.ravel()):
        b = 3*sig[:,i]
        axi.plot(t, delta[:,i], label="μ+ - μ-")
        axi.plot(t, b, "k--", label=f"+3σ {bounds_label}" if i==0 else None)
        axi.plot(t,-b, "k--", label=f"-3σ {bounds_label}" if i==0 else None)
        axi.set_ylabel(labels[i]); axi.grid(True)
    ax[-1,0].set_xlabel("Time (s)"); ax[-1,1].set_xlabel("Time (s)")
    ax[0,0].legend()
    fig.suptitle("Part 4c: State Update Difference within Bounds")
    fig.tight_layout()
    return fig

def plot_residuals(t: np.ndarray, station: np.ndarray, resid: np.ndarray, mask: Optional[np.ndarray]=None) -> plt.Figure:
    if mask is not None:
        t = t[mask]; station = station[mask]; resid = resid[mask]
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,8), sharex=True)
    for s in np.unique(station):
        m = station==s
        ax1.plot(t[m], resid[m,0], ".", label=f"DSN {s}")
        ax2.plot(t[m], resid[m,1], ".", label=f"DSN {s}")
    ax1.axhline(0, linewidth=0.8); ax2.axhline(0, linewidth=0.8)
    ax1.set_ylabel("δρ (km)"); ax2.set_ylabel("δρ̇ (km/s)"); ax2.set_xlabel("Time (s)")
    ax1.grid(True); ax2.grid(True)
    ax1.legend(); ax2.legend()
    fig.suptitle("Part 5a: Post-fit Measurement Residuals")
    fig.tight_layout()
    return fig

def plot_state_with_bounds(t: np.ndarray, x: np.ndarray, P: np.ndarray) -> plt.Figure:
    sig = np.sqrt(np.stack([np.diag(Pk) for Pk in P]))
    labels = ["x (km)","y (km)","z (km)","vx (km/s)","vy (km/s)","vz (km/s)"]
    fig, ax = plt.subplots(3,2,figsize=(12,10), sharex=True)
    for i, axi in enumerate(ax.ravel()):
        b = 3*sig[:,i]
        axi.plot(t, x[:,i], label="μ+" if i==0 else None)
        axi.plot(t, x[:,i]+b, "--", label="+3σ" if i==0 else None)
        axi.plot(t, x[:,i]-b, "--", label="-3σ" if i==0 else None)
        axi.set_ylabel(labels[i]); axi.grid(True)
    ax[-1,0].set_xlabel("Time (s)"); ax[-1,1].set_xlabel("Time (s)")
    ax[0,0].legend()
    fig.suptitle("Part 5c: Estimated State with ±3σ Bounds")
    fig.tight_layout()
    return fig

def residual_report(resid: np.ndarray, mask: Optional[np.ndarray]=None) -> Tuple[float,float]:
    if mask is not None:
        resid = resid[mask]
    rms_r = float(np.sqrt(np.mean(resid[:,0]**2)))
    rms_rd = float(np.sqrt(np.mean(resid[:,1]**2)))
    return rms_r, rms_rd

# -----------------------
# Example run
# -----------------------
def default_x0_from_prompt(kind: Literal["easy","hard"]="easy") -> np.ndarray:
    if kind == "easy":
        a,e,i,om,Om,nu = 7000.0, 0.2, 45*DEG2RAD, 0*DEG2RAD, 270*DEG2RAD, 78.75*DEG2RAD
    else:
        a,e,i,om,Om,nu = 7000.0, 0.6, 45*DEG2RAD, 180*DEG2RAD, 0*DEG2RAD, 45*DEG2RAD
    r0,v0 = coe_to_rv(a,e,i,om,Om,nu)
    return np.hstack((r0,v0))

def main(dataset: Literal["easy","hard"]="easy"):
    fname = "Project-Measurements-Easy.npy" if dataset=="easy" else "Project-Measurements-Hard.npy"
    meas = load_measurements(fname)

    x0 = default_x0_from_prompt(dataset)
    # Tune these if needed:
    pos_sigma_km = 10.0
    vel_sigma_kms = 0.01
    P0 = np.diag([pos_sigma_km**2]*3 + [vel_sigma_kms**2]*3)

    cfg = EKFConfig(
        accel_sigma_kms2=1e-5,
        min_elev_deg=5.0,
        nis_gate=None,     # try 9.21 if you want mild outlier rejection
        joseph_form=True
    )

    plot_measurements(meas)

    # Part 3: prediction-only sigma envelopes (use P_minus from EKF w/ updates disabled)
    pred = run_ekf(meas, x0, P0, EKFConfig(accel_sigma_kms2=cfg.accel_sigma_kms2, min_elev_deg=None))
    plot_sigma_envelopes(pred["t"], pred["P_minus"], "Part 3: Prediction-only ±3σ Covariance Bounds")

    # Full EKF
    out = run_ekf(meas, x0, P0, cfg)
    plot_pre_post_sigma(out["t"], out["P_minus"], out["P_plus"])

    # Part 4c: choose bounds source (pre is most common)
    plot_state_update_vs_bounds(out["t"], out["x_minus"], out["x_plus"], out["P_minus"], bounds_label="pre")

    # Part 5: residuals and state-with-bounds
    plot_residuals(out["t"], out["station"], out["resid"], mask=out["updated"])
    plot_state_with_bounds(out["t"], out["x_plus"], out["P_plus"])

    rms_r, rms_rd = residual_report(out["resid"], out["updated"])
    print(f"Residual RMS: range={rms_r:.6e} km (~{rms_r*1e6:.2f} m), range-rate={rms_rd:.6e} km/s (~{rms_rd*1e5:.2f} cm/s)")
    print("Final state (km, km/s) at t =", out["t"][-1])
    print(out["x_plus"][-1])
    print("Final 1σ:", np.sqrt(np.diag(out["P_plus"][-1])))
    plt.show()

if __name__ == "__main__":
    main("easy")
