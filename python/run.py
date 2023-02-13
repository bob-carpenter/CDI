import datetime
import time
import argparse
from typing import Any

import cmdstanpy
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from pathlib import Path


def rgb2gray(rgb):
    """Convert a nxmx3 RGB array to a grayscale nxm array.

    This function uses the same internal coefficients as MATLAB:
    https://www.mathworks.com/help/matlab/ref/rgb2gray.html
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


REPO_DIR = Path(__file__).parent.parent.resolve()
STAN_FILE = REPO_DIR / "stan" / "holo-cdi-idxs.stan"


def get_data(ura_file: Path, data_file: Path, sigma: float) -> dict[str, Any]:
    R = np.loadtxt(ura_file, delimiter=",", dtype=int)
    N = R.shape[0]
    d = N  # separation - defaults to full size of sample

    data = scio.loadmat(data_file)
    Y_tilde = data["Yt"].astype("int64")
    M1, M2 = Y_tilde.shape
    r = int(data["r"][0, 0])
    N_p = data["Np"][0, 0]

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
    return data


def side_by_side(first, second, data, save_dir=None):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(first[0], cmap="gray", vmin=0, vmax=1)
    ax1.set_title(first[1])
    ax2.imshow(second[0], cmap="gray", vmin=0, vmax=1)
    ax2.set_title(second[1])
    size = data["N"]
    N_p = data["N_p"]
    r = data["r"]
    sigma = data["sigma"]
    fig.suptitle(f"{size}x{size} - N_p: {N_p} - r: {r} - sigma: {sigma}")
    if save_dir is not None:
        fig.savefig(save_dir / f"recovery-{datetime.datetime.now():%Y%m%d%H%M%S}")
    else:
        plt.show()


cli = argparse.ArgumentParser()

cli.add_argument(
    "--size",
    action="store",
    type=str,
    choices=["32", "64", "128", "192", "256"],
    default="256",
)
cli.add_argument(
    "--noise",
    action="store",
    type=str,
    choices=["noiseless", "low_photon", "beamstop", "full"],
    default="full",
)
cli.add_argument(
    "--method",
    action="store",
    type=str,
    choices=["SAMPLE", "OPTIMIZE"],
    default="OPTIMIZE",
)
cli.add_argument("--sigma", type=float, default=1)
cli.add_argument("--init_to_true", action="store_true")
cli.add_argument("--interactive", "-i", action="store_true")
cli.add_argument("--rebuild", action="store_true")


if __name__ == "__main__":

    args = cli.parse_args()

    print(args)

    REF_FILE = REPO_DIR / "data" / args.size / f"URA_{args.size}.csv"
    DATA_FILE = REPO_DIR / "data" / args.size / args.noise / "Yt_data.mat"
    TRUE_IMAGE = REPO_DIR / "data" / args.size / "mimivirus.png"
    data = get_data(REF_FILE, DATA_FILE, args.sigma)
    x_true = rgb2gray(mpimg.imread(TRUE_IMAGE))

    if args.rebuild:
        print("Rebuilding CmdStan")
        cmdstanpy.rebuild_cmdstan(cores=4, progress=True)
    model = cmdstanpy.CmdStanModel(
        stan_file=STAN_FILE, compile="force" if args.rebuild else True
    )

    if args.interactive:
        side_by_side((x_true, "Ground truth"), (data["R"], "reference"), data)
        plt.imshow(np.fft.fftshift(np.log(1 + data["Y_tilde"])), cmap="viridis")
        plt.show()

    RESULT_DIR = (
        Path("/mnt/ceph/users/bward/mcmc/results")
        / f"{args.method}_{args.size}_{args.noise}_{datetime.datetime.now():%Y%m%d%H%M%S}"
    )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("--- START ---")
    if args.method == "OPTIMIZE":
        fit = model.optimize(
            data=data,
            inits={"X": x_true} if args.init_to_true else 1,
            show_console=True,
            save_iterations=True,
            refresh=50,
            output_dir=RESULT_DIR,
        )
        side_by_side(
            (x_true, "True X"),
            (fit.stan_variable("X"), "Recovered X"),
            data,
            RESULT_DIR,
        )
    elif args.method == "SAMPLE":
        before = time.perf_counter()
        fit = model.sample(
            data=data,
            chains=4,
            parallel_chains=4,
            threads_per_chain=1,
            inits={"X": x_true} if args.init_to_true else 1,
            show_console=True,
            output_dir=RESULT_DIR,
            save_warmup=True,
            refresh=10,
            iter_warmup=400,
            iter_sampling=400,
            force_one_process_per_chain=False
        )
        after = time.perf_counter()
        print(f"Sampling took {after - before:0.2f} seconds")
        side_by_side(
            (x_true, "True X"),
            (fit.stan_variable("X").mean(axis=0), "Recovered X"),
            data,
            RESULT_DIR,
        )
