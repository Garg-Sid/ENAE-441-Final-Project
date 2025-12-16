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



def main():

  #plt.show()