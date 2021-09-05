import sys
from Init_MPC import initquadBranchMPC
from MPC_branch import *
from quadruped_branch_dyn import *
import numpy as np
import pdb
import quadruped_env
from utils import Branch_constants, MPCParams

def main():
    # ======================================================================================================================
    # ============================================= Initialize parameters  =================================================
    # ======================================================================================================================
    x0 = np.array([0, 0, 0])        # Initial condition (only for initializing the MPC, not the actual initial state of the sim)
    dt = 0.2
    NB = 2                               # number of branching, 2 means a tree with 1-m-m^2 branches at each level.

    vxm = 0.2
    vym = 0.1
    rm = 0.5
    v0 = 0.2
    n = 3
    d = 3
    N = 25
    L1 = 0.5
    L2 = 1
    W1 = 0.3
    W2 = 0.6
    col_tol = 0.2
    backupcons = [lambda x:backup_forward(x,v0),lambda x:backup_stop(x)]
    cons = Quad_constants(s1=2,s2=3,c2=0.5,alpha=1,R=1.2,vxm=vxm,vym=vym,rm=rm,L1=L1, W1=W1,L2=L2,W2=W2,col_tol = col_tol,col_alpha=5)
    model = PredictiveModel(n, d, N, backupcons, dt, cons)

    # Initialize controller parameters
    xRef = np.array([5.,5.,0.])
    model = PredictiveModel(n, d, N, backupcons, dt, cons)

    mpcParam = initquadBranchMPC(n,d,N,NB,xRef,vxm,vym,rm)
    # mpc = robustMPC(mpcParam, model)
    # mpc = BranchMPC(mpcParam, model)
    mpc = BranchMPCProx(mpcParam, model)

    quadruped_env.sim(mpc)



if __name__== "__main__":
  main()
