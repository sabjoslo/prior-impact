import datetime
import numpy as np
import pickle
import sys
from pybad.tasks.memory_retention import *
from runSimulations import *

pardict = pickle.load(open("par_memreten", "rb"))

dat = -np.ones((2,2,2,3,4,6,100,101))
for i,parest in enumerate([True,False]):
    for j,mlab in enumerate(["pow","exp"]):
        f = [pow_f,exp_f][j]
        if parest:
            design_methods = ["ado_paramest4paramest","fixed4paramest"]
        else:
            design_methods = ["ado_modelest","fixed","ado_fullid"]
        for k,dm in enumerate(design_methods):
            for l,t0 in enumerate(["blow","bhigh","cmpk","uniform"]):
                for m,t1 in enumerate(["blow","bhigh","cmpk","uniform"]):
                    for n in range(100):
                        par = pardict[t0][mlab][n,:]
                        fname = f"{dm}_{mlab}_{t0}_{t1}_{n}"
                        try:
                            res = pickle.load(open(
                                f"{data_dir}/{fname}", "rb"
                            ))
                        except FileNotFoundError:
                            continue
                            
                        sys.stdout.write(
                            f"[{datetime.datetime.now()}]: Reading {fname}.\n"
                        )
                        sys.stdout.flush()
                            
                        pm = np.array(res["pM"])
                        if j == 1 and not parest:
                            pm = 1.-pm
                        dat[0,i,j,k,l,m,n,:] = pm
                        ptheta = [beta(**prior_dict[t1][mlab]).pdf(par).prod()]
                        for o in range(len(res["samples"])-1):
                            if parest:
                                ss = res["samples"][o][0]
                                ww = res["weights"][o][0]
                            else:
                                ss = res["samples"][o][j]
                                ww = res["weights"][o][j]
                            x = res["designs"][o]
                            y = res["Y"][o]
                            pytheta = f(*par, x)
                            py = f(*ss.T, x)@ww
                            if y == 0:
                                pytheta = 1.-pytheta
                                py = 1.-py
                            ptheta.append(ptheta[-1] * pytheta / py)
                        dat[1,i,j,k,l,m,n,:] = ptheta
                        with open("dat_memreten", "wb") as wfh:
                            pickle.dump(dat, wfh)

with open("dat_memreten", "wb") as wfh:
    pickle.dump(dat, wfh)
