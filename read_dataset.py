import numpy as np
import pandas as pd
from scipy.io import loadmat


# ******************************************************************************
# Read in data
# ******************************************************************************
def read_data(path):
    X = pd.read_csv(f'{path}/particle_inputs', delim_whitespace=True, header=None).values
    Y = pd.read_csv(f'{path}/particle_outputs', delim_whitespace=True, header=None).values

    print(X.shape)

    Xsmall = X[:, :700]
    n, _ = Xsmall.shape

    Ysmall = Y[:, :700]
    m, _ = Ysmall.shape

    Xsmall_test = X[:, 700:]
    Ysmall_test = Y[:, 700:]

    # ******************************************************************************
    # Return train and test set
    # ******************************************************************************
    return Xsmall.T, Xsmall_test.T, Ysmall.T, Ysmall_test.T, m, n


def rescale(Xsmall, Xsmall_test):
    # ******************************************************************************
    # Rescale data
    # ******************************************************************************
    Xmin = Xsmall.min()
    Xmax = Xsmall.max()

    Xsmall = ((Xsmall - Xmin) / (Xmax - Xmin))
    Xsmall_test = ((Xsmall_test - Xmin) / (Xmax - Xmin))

    return Xsmall, Xsmall_test
