from math import floor
import scipy.io as scio
import cmdstanpy
import numpy as np
import pathlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def side_by_side(first, second):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(first[0], cmap="gray")
    ax1.set_title(first[1])
    ax2.imshow(second[0], cmap="gray")
    ax2.set_title(second[1])
    plt.show()


SIZE = "32"
NOISE = "low_photon"

REPO_DIR = pathlib.Path(__file__).parent.parent.resolve()

STAN_FILE = REPO_DIR / "stan" / "holo-cdi.stan"
REF_FILE = REPO_DIR / "data" / SIZE / f"URA_{SIZE}.csv"
DATA_FILE = REPO_DIR / "data" / SIZE / NOISE / "Yt_data.mat"
TRUE_IMAGE = REPO_DIR / "data" / SIZE / "mimivirus.png"


model = cmdstanpy.CmdStanModel(stan_file=STAN_FILE)

R = np.loadtxt(REF_FILE, delimiter=",", dtype=int)
N = R.shape[0]

data = scio.loadmat(DATA_FILE)
Y_tilde = data["Yt"].astype("int64")
M1, M2 = Y_tilde.shape
r = int(data["r"][0, 0])
N_p = data["Np"][0, 0]


x_init_true = rgb2gray(mpimg.imread(TRUE_IMAGE))

data = {"N": N, "R": R, "M1": M1, "M2": M2, "Y_tilde": Y_tilde, "r": r, "N_p": N_p}


if __name__ == "__main__":
    print(data)
    # sanity check:
    side_by_side((x_init_true, "Ground truth"), (R, "reference"))

    # fit = model.optimize(
    #     data=data,
    #     inits=1,
    #     # inits={"X": x_init_true},
    #     show_console=True,
    #     save_iterations=True,
    #     output_dir='.'
    # )
    # side_by_side((x_init_true, "True X"), (fit.stan_variable("X"), "Recovered X"))

    before = time.perf_counter()
    fit = model.sample(
        data=data,
        chains=2,
        inits=1,
        # inits={"X": x_init_true},
        show_console=True,
    )
    after = time.perf_counter()
    print(f'Sampling took {after - before:0.2f} seconds')
    fit.save_csvfiles(".")
    side_by_side((x_init_true, "True X"), (fit.stan_variable("X").mean(axis=0), "Recovered X"))
