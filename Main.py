import os
import scipy
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

def load_numpy_data(file_path):
    import os

    cur_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    data = np.load(cur_dir + file_path, allow_pickle=True)
    print(f"Loaded data from {file_path}")
    return data


def part1e_plot_measurements():
    # Load measurement data
    data = load_numpy_data("Project-Measurements-Easy.npy")

    # Extract columns
    t       = data[:, 0]
    idx     = data[:, 1].astype(int)
    rho     = data[:, 2]
    rhodot  = data[:, 3]

    #Figure
    fig = plt.figure(figsize=(10, 8))

    #Range
    ax1 = plt.subplot(2, 1, 1)
    for i in np.unique(idx):
        mask = idx == i
        ax1.plot(t[mask], rho[mask], '.', label=f'DSN {i}')
    ax1.set_ylabel('Range ρ')
    ax1.set_title('Measurements vs Time')
    ax1.grid(True)
    ax1.legend()

    # Range-rate
    ax2 = plt.subplot(2, 1, 2)
    for i in np.unique(idx):
        mask = idx == i
        ax2.plot(t[mask], rhodot[mask], '.', label=f'DSN {i}')
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Range-rate ṙho')
    ax2.grid(True)

    plt.tight_layout()

    return fig




def main():
    fig = part1e_plot_measurements()
    plt.show()


if __name__ == "__main__":
    main()
