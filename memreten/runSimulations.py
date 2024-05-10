#!/usr/bin/env python
# coding: utf-8

"""Set-up design modified from that in Cavagnaro, Myung, and Pitt (2010). 
Adaptive Design Optimization: A Mutual Information-Based Approach to Model
Discrimination in Cognitive Science. *Neural Computation, 22*.
"""

import datetime
import numpy as np
import os
import pickle
from scipy.stats import beta
import sys
from pybad.tasks.memory_retention import *
from pybad.bad.sbinom import *
from pybad.models import *
from pybad.sequential_distributions import *

###### Simulation parameters and helper functions ######

data_dir = "data/"
design_grid = np.arange(100)[:,None]

prior_dict = dict(uniform=dict(
    pow=dict(a=[1.,1.], b=[1.,1.]), exp=dict(a=[1.,1.], b=[1.,1.])
), cmpk=dict(pow=dict(a=[2.,1.], b=[1.,4.]), exp=dict(a=[2.,1.], b=[1.,80.])),
blow=dict(
    pow=dict(a=[1.,1.], b=[1.,2.]), exp=dict(a=[1.,1.], b=[1.,2.])
), bhigh=dict(pow=dict(a=[1.,2.], b=[1.,1.]), exp=dict(a=[1.,2.], b=[1.,1.]))
)

def adaptive_fullIdentification(trialn, models, fixed_design=None):
    return U(design_grid, u_totalEntropy, *models).argmax()

def adaptive_parameterEstimation(trialn, models, fixed_design=None):
    return U(design_grid, u_parameterEstimation, *models).argmax()

def adaptive_modelComparison(trialn, models, fixed_design=None):
    return U(design_grid, u_modelSelection, *models).argmax()

def fixed(trialn, models, fixed_design):
    return fixed_design[trialn % 10]

def reset_models(theta1, sampler=KDE):
    pow_prior = beta(**prior_dict[theta1]["pow"])
    exp_prior = beta(**prior_dict[theta1]["exp"])
    
    bnds = np.array([(0.,1.),(0.,1.)])
    return ( BinaryClassModel(
        f=pow_f, param_bounds=bnds, prior=pow_prior, p_m=.5, dist=sampler
    ), BinaryClassModel(
        f=exp_f, param_bounds=bnds, prior=exp_prior, p_m=.5, dist=sampler
    ) )

def run_experiment(
    par, designf, genf, genf_indx, ntrials, theta1, param_estimation, 
    write=False
):
    fixed_design = np.array([0,1,2,4,7,12,21,35,59,99])
    np.random.shuffle(fixed_design)
    models = reset_models(theta1)
    if param_estimation:
        models = (models[genf_indx],)
        models[0].p_m = 1.
    pM = [models[0].p_m]
    samples = [[m.dist.samples for m in models]]
    W = [[m.dist.W for m in models]]
    D = []
    Y = []
    for ii in range(ntrials):
        d = design_grid[[designf(ii, models, fixed_design=fixed_design)],:]
        y = bernoulli.rvs(p=genf(*par, d))
        update_models(y, d, *models)
        pM.append(models[0].p_m)
        samples.append([m.dist.samples for m in models])
        W.append([m.dist.W for m in models])
        D.append(d)
        Y.append(y)
    data = dict(designs=D, pM=pM, par=par, samples=samples, weights=W, Y=Y)
    if write:
        with open(write, "wb") as wfh:
            pickle.dump(data, wfh)
    return data

###### Main functions ######

# Runs one simulated experiment, where the generating parameter value \theta* is 
# sampled from the population distribution specified by the argument `theta0`, 
# the specified prior is specified by the argument `theta1`, and the sequential
# design method is specified by the argument `samplingfunc`.
def one_sim(iter, theta0, theta1, samplingfunc, generatingfunc, param_est):
    iter = int(iter)
    param_est = {"True": True, "False": False}[param_est]
    
    designf = dict(
        fixed=fixed, ado_fullid=adaptive_fullIdentification, 
        ado_modelest=adaptive_modelComparison, 
        ado_paramest=adaptive_parameterEstimation
    )[samplingfunc]
    if param_est:
        samplingfunc += "4paramest"
    genf = dict(pow=pow_f, exp=exp_f)[generatingfunc]
    genf_indx = dict(pow=0, exp=1)[generatingfunc]
    
    par = pickle.load(open("par_memreten", "rb"))[theta0][
        generatingfunc
    ][iter,:]
      
    fname = f"{data_dir}{samplingfunc}_{generatingfunc}_{theta0}_{theta1}_{iter}"

    if not os.path.exists(fname):
        sys.stdout.write(
            f"[{datetime.datetime.now()}]: Writing to {fname.split('/')[-1]}.\n"
        )
        sys.stdout.flush()

        data = run_experiment(
            par, designf, genf, genf_indx, ntrials=100, theta1=theta1, 
            param_estimation=param_est, write=fname
        )

    else:
        sys.stdout.write(
            f"[{datetime.datetime.now()}]: Path {fname.split('/')[-1]} exists.\n"
        )
        sys.stdout.flush()

# Outputs a list of arguments that be can iteratively passed to `one_sim()` to
# reproduce all results reported in the manuscript.
def get_args():
    args_paramest = np.meshgrid(
        np.arange(100), ["blow","bhigh"], ["uniform","cmpk","blow","bhigh"],
        ["ado_paramest","fixed"], ["pow"], [True]
    )
    args_paramest = np.vstack([ x.ravel() for x in args_paramest ])
    args_modelest = np.meshgrid(
        np.arange(100), ["uniform","cmpk"], ["uniform","cmpk"], 
        ["ado_modelest","ado_fullid","fixed"], ["pow","exp"], [False]
    )
    args_modelest = np.vstack([ x.ravel() for x in args_modelest ])
    return np.hstack((args_paramest,args_modelest))
