from math import floor
import scipy.io as scio
import cmdstanpy
import numpy as np
import pathlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


REPO_DIR = pathlib.Path(__file__).parent.parent.resolve()

STAN_FILE = REPO_DIR / "stan" / "holo-cdi.stan"
REF_FILE = REPO_DIR / "data" / "URA_256.csv"
DATA_FILE = REPO_DIR / "data" / "Yt_data.mat"
TRUE_IMAGE = REPO_DIR / "data" / "mimivirus.png"


model = cmdstanpy.CmdStanModel(stan_file=STAN_FILE)

R = np.loadtxt(REF_FILE, delimiter=",", dtype=int)
N = R.shape[0]

data = scio.loadmat(DATA_FILE)
Y_tilde = data["Yt"].astype("int64")
M1, M2 = Y_tilde.shape
r = int(data["r"][0, 0])
N_p = data["Np"][0, 0]

# more realistic values
# r = floor(0.5*0.05*np.mean([M1,M2]))
# N_p = 10 # ???

x_init_true = rgb2gray(mpimg.imread(TRUE_IMAGE))
# sanity check:
# plt.imshow(x_init_true, cmap="gray")
# plt.show()

data = {"N": N, "R": R, "M1": M1, "M2": M2, "Y_tilde": Y_tilde, "r": r, "N_p": N_p}

if __name__ == "__main__":
    fit = model.optimize(data=data, show_console=True, inits=1)  # {'X': x_init_true}
    fit.save_csvfiles(".")
    plt.imshow(fit.stan_variable("X"), cmap="gray")
    plt.show()
