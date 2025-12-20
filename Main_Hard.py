from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import Main as base


BASE_POS_SIGMA_KM = 10.0
BASE_VEL_SIGMA_KMS = 0.05
NIS_GATE_997 = 11.83  # Chi-square 99.7% for 2 dof.
NIS_95_LO = 0.0506
NIS_95_HI = 7.3778


@dataclass(frozen=True)
class HardConfig:
    state_sigma_scale: float = 1.0
    accel_noise_std: float = 1e-5
    elevation_mask_deg: Optional[float] = None
    nis_gate: Optional[float] = None


@dataclass(frozen=True)
class ScoreWeights:
    rms: float = 1.0
    nis: float = 1.0
    mean: float = 0.2
    accept: float = 5.0
    drift: float = 0.5
    station_spread: float = 0.3
    nis_coverage: float = 0.2


@dataclass(frozen=True)
class MetricBundle:
    sigma_r: float
    sigma_rdot: float
    rms_range: float
    rms_rdot: float
    mean_range: float
    mean_rdot: float
    nis_mean: float
    nis_95_frac: float
    accept_rate: float
    drift_ratio: float
    station_rms: Dict[int, float]
    reject_counts: Dict[str, int]

    def score(self, weights: Optional[ScoreWeights] = None) -> float:
        if weights is None:
            weights = ScoreWeights()
        rms_range = self.rms_range / self.sigma_r
        rms_rdot = self.rms_rdot / self.sigma_rdot
        mean_range = self.mean_range / self.sigma_r
        mean_rdot = self.mean_rdot / self.sigma_rdot
        nis_dev = abs(self.nis_mean - 2.0)
        nis_cov = abs(self.nis_95_frac - 0.95)
        drift = abs(self.drift_ratio - 1.0)
        station_spread = 0.0
        if len(self.station_rms) > 1:
            values = np.array(list(self.station_rms.values()))
            station_spread = float(np.max(values) / np.min(values) - 1.0)

        score = 0.0
        score += weights.rms * (abs(rms_range - 1.0) + abs(rms_rdot - 1.0))
        score += weights.nis * nis_dev
        score += weights.mean * (abs(mean_range) + abs(mean_rdot))
        score += weights.drift * drift
        score += weights.station_spread * station_spread
        score += weights.nis_coverage * nis_cov
        if self.accept_rate < 0.7:
            score += weights.accept * (0.7 - self.accept_rate)
        return float(score)

    def summary(self) -> str:
        return (
            "RMS r {:.2e}, RMS rdot {:.2e}, NIS {:.2f}, "
            "NIS95 {:.2f}, accept {:.2f}".format(
                self.rms_range,
                self.rms_rdot,
                self.nis_mean,
                self.nis_95_frac,
                self.accept_rate,
            )
        )


@dataclass(frozen=True)
class Trial:
    label: str
    config: HardConfig
    metrics: MetricBundle
    score: float


def part7_initial_conditions(
    state_sigma_scale: float = 1.0,
    accel_noise_std: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Part 7 initial guess for the harder dataset.
    Uses the provided classical elements and same noise assumptions as Part 3/5.
    """
    a = 7000.0
    e = 0.6
    inc = 45.0 * base.DEG2RAD
    arg_perigee = 180.0 * base.DEG2RAD
    raan = 0.0 * base.DEG2RAD
    true_anomaly = 45.0 * base.DEG2RAD

    r0, v0 = base.coe_to_cartesian_state(
        a, e, inc, arg_perigee, raan, true_anomaly
    )
    x0 = np.hstack((r0, v0))
    pos_sigma_km = BASE_POS_SIGMA_KM * state_sigma_scale
    vel_sigma_kms = BASE_VEL_SIGMA_KMS * state_sigma_scale
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
    R = base.measurement_noise_matrix()
    return x0, P0, accel_noise_std, R


def run_ekf_hard(
    data: np.ndarray,
    config: HardConfig,
    apply_measurement_updates: bool = True,
) -> Dict[str, object]:
    """
    EKF loop for the hard dataset using Part 7 initial conditions.
    Mirrors the Main.run_ekf logic but seeds with the Part 7 orbit.
    """
    times = data[:, 0]
    stations = data[:, 1].astype(int)
    measurements = data[:, 2:4]

    x0, P0, accel_sigma, R = part7_initial_conditions(
        state_sigma_scale=config.state_sigma_scale,
        accel_noise_std=config.accel_noise_std,
    )

    N = len(times)
    n = len(x0)

    x_minus = np.zeros((N, n))
    P_minus = np.zeros((N, n, n))
    x_plus = np.zeros((N, n))
    P_plus = np.zeros((N, n, n))

    innovations = np.zeros((N, 2))
    residuals = np.zeros((N, 2))
    S_all = np.zeros((N, 2, 2))
    nis = np.zeros(N)
    accepted = np.ones(N, dtype=bool)
    rejected_elevation = 0
    rejected_nis = 0

    x = x0.copy()
    P = P0.copy()
    t_prev = times[0]

    I = np.eye(n)

    for k, (tk, station, meas) in enumerate(zip(times, stations, measurements)):
        dt = tk - t_prev if k > 0 else 0.0

        if dt > 0.0:
            x, Phi = base.discrete_state_transition(t_prev, tk, x)
            Qk = base.process_noise_matrix(dt, accel_sigma)
            P = Phi @ P @ Phi.T + Qk

        x_minus[k, :] = x
        P_minus[k, :, :] = P

        y_hat_minus = base.measurement_function(x, tk, station)
        Hk = base.measurement_jacobian(x, tk, station)
        Sk = Hk @ P @ Hk.T + R
        S_all[k, :, :] = Sk

        nu = meas - y_hat_minus
        innovations[k, :] = nu

        nis_k = float(nu.T @ np.linalg.solve(Sk, nu))
        nis[k] = nis_k

        accept = True
        if config.elevation_mask_deg is not None:
            elev_deg = float(base.elevation_angle(x, tk, station) * 180.0 / np.pi)
            if elev_deg < config.elevation_mask_deg:
                accept = False
                rejected_elevation += 1

        if config.nis_gate is not None and nis_k > config.nis_gate:
            accept = False
            rejected_nis += 1

        if apply_measurement_updates and accept:
            Kk = P @ Hk.T @ np.linalg.solve(Sk, np.eye(2))
            x = x + Kk @ nu
            P = (I - Kk @ Hk) @ P @ (I - Kk @ Hk).T + Kk @ R @ Kk.T
        else:
            accepted[k] = False

        x_plus[k, :] = x
        P_plus[k, :, :] = P

        y_hat_plus = base.measurement_function(x, tk, station)
        residuals[k, :] = meas - y_hat_plus

        t_prev = tk

    return {
        "times": times,
        "stations": stations,
        "x_minus": x_minus,
        "P_minus": P_minus,
        "x_plus": x_plus,
        "P_plus": P_plus,
        "innovations": innovations,
        "residuals": residuals,
        "S": S_all,
        "nis": nis,
        "accepted": accepted,
        "reject_counts": {
            "elevation": rejected_elevation,
            "nis": rejected_nis,
        },
        "config": config,
    }


def summarize_measurements(data: np.ndarray) -> None:
    times = data[:, 0]
    stations = data[:, 1].astype(int)
    dt = np.diff(times)
    print("\nHard Dataset Summary")
    print(f"  Measurements: {len(times)}")
    print(f"  Time span: {times[0]:.1f} to {times[-1]:.1f} s")
    if len(dt) > 0:
        print(
            "  Gap stats (s): min {:.2f}, median {:.2f}, max {:.2f}".format(
                float(np.min(dt)),
                float(np.median(dt)),
                float(np.max(dt)),
            )
        )
    for station in np.unique(stations):
        count = int(np.sum(stations == station))
        print(f"  Station {station} count: {count}")


def _safe_ratio(numerator: float, denominator: float, default: float = 1.0) -> float:
    if denominator == 0.0 or not np.isfinite(denominator):
        return default
    return float(numerator / denominator)


def _window_mask(total: int, fraction: float, tail: bool = False) -> np.ndarray:
    if total <= 0:
        return np.zeros(0, dtype=bool)
    count = max(1, int(total * fraction))
    idx = np.arange(total)
    if tail:
        return idx >= (total - count)
    return idx < count


def suggest_accel_noise(
    current: float, nis_mean: float, target: float = 2.0
) -> float:
    if not np.isfinite(nis_mean) or nis_mean <= 0.0:
        return current
    scale = np.sqrt(target / nis_mean)
    scale = float(np.clip(scale, 0.3, 3.0))
    return current * scale


def update_leaderboard(
    leaderboard: List[Trial], trial: Trial, max_items: int = 3
) -> List[Trial]:
    updated = sorted(leaderboard + [trial], key=lambda item: item.score)
    return updated[:max_items]


def evaluate_ekf_results(
    results: Dict[str, object], R: np.ndarray
) -> MetricBundle:
    residuals = results["residuals"]
    innovations = results["innovations"]
    S_all = results["S"]
    accepted = results.get("accepted", np.ones(len(residuals), dtype=bool))
    stations = results["stations"]

    sigma_r = float(np.sqrt(R[0, 0]))
    sigma_rdot = float(np.sqrt(R[1, 1]))

    if np.any(accepted):
        rms_range = float(np.sqrt(np.mean(residuals[accepted, 0] ** 2)))
        rms_rdot = float(np.sqrt(np.mean(residuals[accepted, 1] ** 2)))
        mean_range = float(np.mean(residuals[accepted, 0]))
        mean_rdot = float(np.mean(residuals[accepted, 1]))
    else:
        rms_range = float("inf")
        rms_rdot = float("inf")
        mean_range = 0.0
        mean_rdot = 0.0

    nis_vals = results.get("nis", np.zeros(len(innovations)))
    nis_mean = float(np.mean(nis_vals[accepted])) if np.any(accepted) else float("inf")
    nis_95_frac = (
        float(np.mean((nis_vals[accepted] >= NIS_95_LO) & (nis_vals[accepted] <= NIS_95_HI)))
        if np.any(accepted)
        else 0.0
    )
    accept_rate = float(np.mean(accepted))

    station_rms = {}
    for station in np.unique(stations):
        mask = (stations == station) & accepted
        if np.any(mask):
            station_rms[int(station)] = float(
                np.sqrt(np.mean(residuals[mask, 0] ** 2))
            )

    total = len(residuals)
    head_mask = accepted & _window_mask(total, 0.25, tail=False)
    tail_mask = accepted & _window_mask(total, 0.25, tail=True)
    rms_head = float(np.sqrt(np.mean(residuals[head_mask, 0] ** 2))) if np.any(head_mask) else 0.0
    rms_tail = float(np.sqrt(np.mean(residuals[tail_mask, 0] ** 2))) if np.any(tail_mask) else 0.0
    drift_ratio = _safe_ratio(rms_tail, rms_head, default=1.0)

    reject_counts = results.get("reject_counts", {"elevation": 0, "nis": 0})

    return MetricBundle(
        sigma_r=sigma_r,
        sigma_rdot=sigma_rdot,
        rms_range=rms_range,
        rms_rdot=rms_rdot,
        mean_range=mean_range,
        mean_rdot=mean_rdot,
        nis_mean=nis_mean,
        nis_95_frac=nis_95_frac,
        accept_rate=accept_rate,
        drift_ratio=drift_ratio,
        station_rms=station_rms,
        reject_counts=reject_counts,
    )


def score_metrics(metrics: MetricBundle, weights: Optional[ScoreWeights] = None) -> float:
    return metrics.score(weights)


def print_metrics(label: str, metrics: MetricBundle, score: Optional[float] = None) -> None:
    suffix = ""
    if score is not None:
        suffix = f", score {score:.2f}"
    reject_info = ""
    if metrics.reject_counts:
        reject_info = " | rej elev {}, rej nis {}".format(
            metrics.reject_counts.get("elevation", 0),
            metrics.reject_counts.get("nis", 0),
        )
    print(f"  {label} | {metrics.summary()}{suffix}{reject_info}")


def run_tuning_study(
    data: np.ndarray,
) -> Tuple[Dict[str, object], HardConfig, MetricBundle]:
    base_config = HardConfig()
    R = base.measurement_noise_matrix()
    weights = ScoreWeights()
    leaderboard: List[Trial] = []

    print("\nHard Dataset Tuning Study")
    baseline_results = run_ekf_hard(data, base_config)
    baseline_metrics = evaluate_ekf_results(baseline_results, R)
    baseline_score = baseline_metrics.score(weights)
    baseline_trial = Trial("baseline", base_config, baseline_metrics, baseline_score)
    print_metrics("baseline", baseline_metrics, baseline_score)

    best_trial = baseline_trial
    best_results = baseline_results
    leaderboard = update_leaderboard(leaderboard, baseline_trial)

    def try_config(label: str, config: HardConfig) -> Trial:
        nonlocal best_trial, best_results, leaderboard
        results = run_ekf_hard(data, config)
        metrics = evaluate_ekf_results(results, R)
        score = metrics.score(weights)
        trial = Trial(label, config, metrics, score)
        print_metrics(label, metrics, score)
        leaderboard = update_leaderboard(leaderboard, trial)
        if score < best_trial.score:
            best_trial = trial
            best_results = results
        return trial

    print("Step 1: adapt accel noise toward NIS target")
    accel = base_config.accel_noise_std
    for i in range(2):
        trial = try_config(f"accel iter {i + 1}", replace(base_config, accel_noise_std=accel))
        accel = suggest_accel_noise(accel, trial.metrics.nis_mean)
    accel_candidates = [accel / 2.0, accel, accel * 2.0]
    for accel in accel_candidates:
        try_config(f"accel {accel:.2e}", replace(base_config, accel_noise_std=accel))
    base_config = replace(base_config, accel_noise_std=best_trial.config.accel_noise_std)

    print("Step 2: sweep initial covariance scale")
    for scale in [0.5, 1.0, 2.0, 3.0]:
        try_config(f"scale {scale:.1f}", replace(base_config, state_sigma_scale=scale))
    base_config = replace(base_config, state_sigma_scale=best_trial.config.state_sigma_scale)

    print("Step 3: test robust update strategies")
    strategies = [
        ("baseline", None, None),
        ("elevation", 10.0, None),
        ("nis_gate", None, NIS_GATE_997),
        ("both", 10.0, NIS_GATE_997),
    ]
    for label, elev_mask, nis_gate in strategies:
        cfg = replace(
            base_config,
            elevation_mask_deg=elev_mask,
            nis_gate=nis_gate,
        )
        try_config(label, cfg)
    base_config = best_trial.config

    print("Step 4: local refinement around best")
    local_scales = sorted(
        {base_config.state_sigma_scale * f for f in [0.7, 1.0, 1.3]}
    )
    local_accels = sorted(
        {base_config.accel_noise_std * f for f in [0.7, 1.0, 1.3]}
    )
    for scale in local_scales:
        for accel in local_accels:
            cfg = replace(
                base_config,
                state_sigma_scale=scale,
                accel_noise_std=accel,
            )
            try_config(f"refine s{scale:.2f} a{accel:.2e}", cfg)

    print("\nTop configs:")
    for idx, trial in enumerate(leaderboard, start=1):
        print(f"  {idx}. {trial.label} | {trial.metrics.summary()} | score {trial.score:.2f}")

    print(
        "Selected config: scale {:.2f}, accel {:.2e}, elev {}, nis {}".format(
            best_trial.config.state_sigma_scale,
            best_trial.config.accel_noise_std,
            best_trial.config.elevation_mask_deg,
            best_trial.config.nis_gate,
        )
    )
    return best_results, best_trial.config, best_trial.metrics


def main() -> None:
    data = base.load_numpy_data("Project-Measurements-Hard.npy")

    summarize_measurements(data)

    ekf_results, best_config, best_metrics = run_tuning_study(data)

    print(
        "\nBest metrics: RMS r {:.2e} km, RMS rdot {:.2e} km/s, "
        "NIS {:.2f}".format(
            best_metrics.rms_range,
            best_metrics.rms_rdot,
            best_metrics.nis_mean,
        )
    )

    # Plot only the most informative debugging figures.
    base.part5_plot_residuals(
        ekf_results["times"],
        ekf_results["stations"],
        ekf_results["residuals"],
    )

    base.part5_plot_state_with_bounds(
        ekf_results["times"],
        ekf_results["x_plus"],
        ekf_results["P_plus"],
    )

    base.part5_report_final_state(
        ekf_results["times"],
        ekf_results["x_plus"],
        ekf_results["P_plus"],
    )

    base.ekf_performance_tests(
        ekf_results,
        base.measurement_noise_matrix(),
    )

    plt.show()


if __name__ == "__main__":
    main()
