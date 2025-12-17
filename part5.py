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