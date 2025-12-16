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


def propogate_CT_LTI_analytically(X_0, A, t_vec):
    # Return trajectory over time where np.shape(X_t) = (len(t_vec), len(X_0))
    X_0 = np.asarray(X_0, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float)
    t_vec = np.asarray(t_vec, dtype=float)
    n = X_0.size
    t0 = float(t_vec[0])

    X_t = np.empty((t_vec.size, n), dtype=float)
    for i, t in enumerate(t_vec):
        Phi = scipy.linalg.expm(A * (t - t0))
        Phi_sq = np.matmul(Phi, Phi)  # STM squared
        X_t[i] = Phi_sq @ X_0

    if n >= 2:
        plt.figure()
        plt.plot(t_vec, X_t[:, 0], label="x(t)")
        plt.plot(t_vec, X_t[:, 1], label="ẋ(t)")
        plt.xlabel("t")
        plt.ylabel("state")
        plt.title("CT LTI (analytic, STM²): displacement & velocity")
        plt.legend()
        plt.grid(True)
        plt.show()
    return X_t



def main():
    propogate_CT_LTI_analytically
    #plt.show()

if __name__ == ("__main"
                "__"):