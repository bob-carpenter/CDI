import datetime
import pathlib
import time

import cmdstanpy
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio


def rgb2gray(rgb):
    """Convert a nxmx3 RGB array to a grayscale nxm array.

    This function uses the same internal coefficients as MATLAB:
    https://www.mathworks.com/help/matlab/ref/rgb2gray.html
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


SIZE = "256"  # 32, 64, 128, 192, 256
NOISE = "full"  # noiseless, low_photon, beamstop, full
METHOD = "SAMPLE"  # OPTIMIZE, SAMPLE
INIT_TO_TRUE = False

REPO_DIR = pathlib.Path(__file__).parent.parent.resolve()

STAN_FILE = REPO_DIR / "stan" / "holo-cdi-idxs.stan"
REF_FILE = REPO_DIR / "data" / SIZE / f"URA_{SIZE}.csv"
DATA_FILE = REPO_DIR / "data" / SIZE / NOISE / "Yt_data.mat"
TRUE_IMAGE = REPO_DIR / "data" / SIZE / "mimivirus.png"
RESULT_DIR = REPO_DIR / "results" / METHOD / SIZE / NOISE
RESULT_DIR.mkdir(parents=True, exist_ok=True)

model = cmdstanpy.CmdStanModel(
    stan_file=STAN_FILE,
    # stanc_options={"O1": True},
)

R = np.loadtxt(REF_FILE, delimiter=",", dtype=int)
N = R.shape[0]
d = N  # separation - defaults to full size of sample

data = scio.loadmat(DATA_FILE)
Y_tilde = data["Yt"].astype("int64")
M1, M2 = Y_tilde.shape
r = int(data["r"][0, 0])
N_p = data["Np"][0, 0]

sigma = 1  # smoothing

data = {
    "N": N,
    "R": R,
    "d": d,
    "M1": M1,
    "M2": M2,
    "Y_tilde": Y_tilde,
    "r": r,
    "N_p": N_p,
    "sigma": sigma,
}

x_true = rgb2gray(mpimg.imread(TRUE_IMAGE))


def side_by_side(first, second, save=False):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(first[0], cmap="gray", vmin=0, vmax=1)
    ax1.set_title(first[1])
    ax2.imshow(second[0], cmap="gray", vmin=0, vmax=1)
    ax2.set_title(second[1])
    fig.suptitle(
        f"{SIZE}x{SIZE} - N_p: {N_p} - r: {r} - sigma: {sigma} - init: {'true image' if INIT_TO_TRUE else '1'}"
    )
    if save:
        fig.savefig(RESULT_DIR / f"recovery-{datetime.datetime.now():%Y%m%d%H%M%S}")
    plt.show()


if __name__ == "__main__":
    # sanity check:
    print(data)
    side_by_side((x_true, "Ground truth"), (R, "reference"))
    # show input frequencies
    plt.imshow(np.fft.fftshift(np.log(1 + Y_tilde)), cmap="viridis")
    plt.show()

    if METHOD == "OPTIMIZE":
        fit = model.optimize(
            data=data,
            inits={"X": x_true} if INIT_TO_TRUE else 1,
            show_console=True,
            save_iterations=True,
            output_dir=RESULT_DIR,
        )
        side_by_side((x_true, "True X"), (fit.stan_variable("X"), "Recovered X"), True)
    elif METHOD == "SAMPLE":
        before = time.perf_counter()
        fit = model.sample(
            data=data,
            chains=2,
            inits={"X": x_true} if INIT_TO_TRUE else 1,
            show_console=True,
            output_dir=RESULT_DIR,
            save_warmup=True,
            refresh=10,
            iter_warmup=400,
            iter_sampling=400,
        )
        after = time.perf_counter()
        print(f"Sampling took {after - before:0.2f} seconds")
        side_by_side(
            (x_true, "True X"),
            (fit.stan_variable("X").mean(axis=0), "Recovered X"),
            True,
        )
