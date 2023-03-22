#!/usr/bin/env python
# coding: utf-8

import datetime
import numpy as np
import os
import pickle
from scipy.stats import bernoulli, norm
import sys
from pybad.bad.sbinom import *
from pybad.models import *

###### Simulation parameters and helper functions ######

data_dir = "data/"
design_methods = ["ado_paramest","fixed"]
Theta0 = ["norm_minus2_1","norm_0_1","norm_2_1"]
Theta1 = ["norm_0_point65","norm_0_1","norm_0_2","norm_2_1"]

design_grid = np.linspace(-3., 3., 31)[:,None]

# Item-response function from Weiss, D. J., & McBride, J. R. (1983). *Bias and 
# information of bayesian adaptive testing* (Research Report No. 83-2). 
# Minneapolis, MN: Computerized Adaptive Testing Laboratory, Department of 
# Psychology, University of Minnesota.
def f(theta, x):
    return .2 + .8 / (1. + np.exp(-2.72*(theta - x[:,0])))

prior_dict = dict(
    norm_minus2_1=dict(loc=-2., scale=1.), 
    norm_0_point65=dict(loc=0., scale=.65), norm_0_1=dict(loc=0., scale=1.), 
    norm_0_2=dict(loc=0., scale=2.), norm_2_1=dict(loc=2., scale=1.)
)

def adaptive_parameterEstimation(trialn, model, fixed_design=None):
    return design_grid[
        [U(design_grid, u_parameterEstimation, model).argmax()],:
    ]

def fixed(trialn, model, fixed_design):
    return fixed_design[[trialn],None]

def reset_model(theta1):
    prior = norm(**prior_dict[theta1])
    return BinaryClassModel(f=f, prior=prior, dist=Gaussian, scale_cov=1.5)

def run_experiment(par, designf, ntrials, theta1, write=False):
    fixed_design = np.linspace(-3., 3., 31)
    np.random.shuffle(fixed_design)
    model = reset_model(theta1)
    samples = [model.dist.samples]
    W = [model.dist.W]
    D = []
    Y = []
    for ii in range(ntrials):
        d = designf(ii, model, fixed_design=fixed_design)
        y = bernoulli.rvs(p=f(*par, d))
        update_models(y, d, model)
        samples.append(model.dist.samples)
        W.append(model.dist.W)
        D.append(d)
        Y.append(y)
    data = dict(designs=D, par=par, samples=samples, weights=W, Y=Y)
    if write:
        with open(write, "wb") as wfh:
            pickle.dump(data, wfh)
    return data

###### Main functions ######

# Runs one simulated experiment, where the generating parameter value \theta* is 
# sampled from the population distribution specified by the argument `theta0`, 
# the specified prior is specified by the argument `theta1`, and the sequential
# design method is specified by the argument `samplingfunc`.
def one_sim(iter, theta0, theta1, samplingfunc):
    iter = int(iter)
    
    designf = dict(fixed=fixed, ado_paramest=adaptive_parameterEstimation)[
        samplingfunc
    ]
    
    par = pickle.load(open("par_irt", "rb"))[theta0][iter,:]
      
    fname = f"{data_dir}{samplingfunc}_{theta0}_{theta1}_{iter}"

    if not os.path.exists(fname):

        sys.stdout.write(
            f"[{datetime.datetime.now()}]: Writing to {fname.split('/')[-1]}.\n"
        )
        sys.stdout.flush()

        data = run_experiment(
            par, designf, ntrials=31, theta1=theta1, write=fname
        )

    else:
        sys.stdout.write(
            f"[{datetime.datetime.now()}]: Path {fname.split('/')[-1]} exists.\n"
        )
        sys.stdout.flush()

# Outputs a list of arguments that be can iteratively passed to `one_sim()` to
# reproduce all results reported in the manuscript.
def get_args():
    args = np.meshgrid(np.arange(1000), Theta0, Theta1, design_methods)
    args = np.vstack([ x.ravel() for x in args ])
    args = args[:,~( (args[1,:] == "norm_minus2_1") & (args[2,:] != "norm_0_1") )]
    args = args[:,~( (args[1,:] == "norm_0_1") & (args[2,:] == "norm_2_1") )]
    args = args[:,~( (args[1,:] == "norm_2_1") & (args[2,:] == "norm_0_point65") )]
    return args