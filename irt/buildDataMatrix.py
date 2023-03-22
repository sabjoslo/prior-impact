import datetime
import numpy as np
import pickle
from scipy.stats import norm
import sys
from runSimulations import *

pardict = pickle.load(open("par_irt", "rb"))

dat = -np.ones((2,len(Theta0),len(Theta1),1000,32))
for i,dm in enumerate(design_methods):
    for j,t0 in enumerate(Theta0):
        for k,t1 in enumerate(Theta1):
            for l in range(1000):
                fname = f"{dm}_{t0}_{t1}_{l}"
                
                par = pardict[t0][l,:]
                try:
                    res = pickle.load(open(f"{data_dir}/{fname}", "rb"))
                except FileNotFoundError:
                    sys.stdout.write(
                        f"[{datetime.datetime.now()}]: {fname} not found.\n"
                    )
                    sys.stdout.flush()
                    continue
                            
                sys.stdout.write(
                    f"[{datetime.datetime.now()}]: Reading {fname}.\n"
                )
                sys.stdout.flush()
                
                # Posterior likelihood of \theta^*
                ptheta = [norm(**prior_dict[t1]).pdf(par).prod()]
                for t in range(len(res["samples"])-1):
                    ss = res["samples"][t]
                    ww = res["weights"][t]
                    x = res["designs"][t]
                    y = res["Y"][t]
                    pytheta = f(*par, x)[0]
                    py = f(*ss.T, x)@ww
                    if y == 0:
                        pytheta = 1.-pytheta
                        py = 1.-py
                    ptheta.append(ptheta[-1] * pytheta / py)
                dat[i,j,k,l,:] = ptheta
                
                with open("dat_irt", "wb") as wfh:
                    pickle.dump(dat, wfh)

with open("dat_irt", "wb") as wfh:
    pickle.dump(dat, wfh)
