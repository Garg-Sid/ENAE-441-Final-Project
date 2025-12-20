import numpy as np
import matplotlib.pyplot as plt

import Main as base


def part7_initial_conditions(
    pos_sigma_km: float = 10.0,
    vel_sigma_kms: float = 0.05,
    accel_noise_std: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
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

    r0, v0 = base.coe_to_cartesian_state(a, e, inc, arg_perigee, raan, true_anomaly)
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
    R = base.measurement_noise_matrix()
    return x0, P0, accel_noise_std, R


def run_ekf_hard(
    data: np.ndarray,
    apply_measurement_updates: bool = True,
) -> dict[str, np.ndarray]:
    """
    EKF loop for the hard dataset using Part 7 initial conditions.
    Mirrors the Main.run_ekf logic but seeds with the Part 7 orbit.
    """
    times = data[:, 0]
    stations = data[:, 1].astype(int)
    measurements = data[:, 2:4]

    x0, P0, accel_sigma, R = part7_initial_conditions()

    N = len(times)
    n = len(x0)

    x_minus = np.zeros((N, n))
    P_minus = np.zeros((N, n, n))
    x_plus = np.zeros((N, n))
    P_plus = np.zeros((N, n, n))

    innovations = np.zeros((N, 2))
    residuals = np.zeros((N, 2))
    S_all = np.zeros((N, 2, 2))

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

        if apply_measurement_updates:
            Kk = P @ Hk.T @ np.linalg.solve(Sk, np.eye(2))
            x = x + Kk @ nu
            P = (I - Kk @ Hk) @ P @ (I - Kk @ Hk).T + Kk @ R @ Kk.T

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
    }


def main() -> None:
    data = base.load_numpy_data("Project-Measurements-Hard.npy")

    base.part1e_plot_measurements(data)

    x0, P0, accel_sigma, _ = part7_initial_conditions()
    fig = base.part5c_plot_prediction_bounds_smooth(data[:, 0], x0, P0, accel_sigma)
    fig.suptitle("Part 7: Prediction-only ±3σ Bounds (Hard Dataset)")

    ekf_results = run_ekf_hard(data, apply_measurement_updates=True)

    base.part4_plot_pre_post_covariance(
        ekf_results["times"],
        ekf_results["P_minus"],
        ekf_results["P_plus"],
    )

    base.part4_plot_state_difference(
        ekf_results["times"],
        ekf_results["x_minus"],
        ekf_results["x_plus"],
        ekf_results["P_minus"],
    )

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
