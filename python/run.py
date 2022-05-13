from math import floor
import cmdstanpy
import numpy as np
import pathlib


REPO_DIR =   pathlib.Path(__file__).parent.parent.resolve()

STAN_FILE = REPO_DIR / "stan" / "holo-cdi.stan"
REF_FILE = REPO_DIR / "data" / "URA_256.csv"
DATA_FILE = REPO_DIR / "data" / "Y_tilde.csv"


model = cmdstanpy.CmdStanModel(stan_file=STAN_FILE)


R = np.loadtxt(REF_FILE, delimiter=',', dtype=int)
N = R.shape[0]
Y_tilde = np.loadtxt(DATA_FILE, delimiter=',', dtype=int)
M1, M2 = Y_tilde.shape
r = 0
N_p = 1e4
# more realistic values
# r = floor(0.5*0.05*np.mean([M1,M2]))
# N_p = 10 # ???

data = {"N":N, "R":R, "M1":M1, "M2":M2, "Y_tilde":Y_tilde, "r":r, "N_p":N_p}

if __name__ == "__main__":
    fit = model.sample(data=data, chains=1, show_console=True, inits=0)
