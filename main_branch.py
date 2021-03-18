import sys
import matplotlib.pyplot as plt
from Init_MPC import initBranchMPC
from MPC_branch import BranchMPC, BranchMPCParams
from highway_branch_dyn import *
import Highway_env
import numpy as np
import pickle
import pdb
import Highway_env_branch

def main():
    # ======================================================================================================================
    # ============================================= Initialize parameters  =================================================
    # ======================================================================================================================
    N = 8
    n = 4;   d = 2                            # State and Input dimension
    x0 = np.array([0, 1.8, 0, 0])        # Initial condition
    am = 6.0
    rm = 0.3
    dt = 0.1
    NB = 2

    N_lane = 4


    # Initialize controller parameters
    xRef = np.array([0.5,1.8,15,0])
    cons =  HMM_constants(s1=2,s2=3,c2=0.5,tran_diag=0.3,alpha=1,R=1.2,am=am,rm=rm,J_c=20,s_c=1,ylb = 0.,yub = 7.2,L=4, W=2.5,col_alpha=5,Kpsi = 0.5)
    backupcons = [lambda x:backup_maintain(x,cons),lambda x:backup_brake(x,cons),lambda x:backup_lc(x,xRef)]
    # backupcons = [lambda x:backup_maintain(x,cons),lambda x:backup_brake(x,cons)]
    model = PredictiveModel(n, d, N, backupcons, dt, cons)
    # xuc = np.array([[0,0,1,0],[6,3,1,0],[-5,3,1,0.1]])
    x = np.array([0.5,0.6,1,-0.1])
    z = np.array([8.5,0.6,1,0])
    u = np.array([3,0.3])
    # p,dp = model.branch_eval(x,z)
    # zpred = model.zpred_eval(z)
    # A,B,C,xp = model.dyn_linearization(x,u)
    # h,dh = model.col_eval(x,z)


    mpcParam = initBranchMPC(n,d,N,NB,xRef,am,rm,N_lane,cons.W)
    mpc = BranchMPC(mpcParam, model)
    # mpc.solve(x,z,xRef)
    # mpc.solve(x,z,xRef)
    # pdb.set_trace()
    # # Init simulators
    # simulator     = Simulator(map)

    # ======================================================================================================================
    # ======================================= PID path following ===========================================================
    # ======================================================================================================================


    # ======================================================================================================================
    # ======================================  LINEAR REGRESSION ============================================================
    # ======================================================================================================================

    # mpc = MPC(mpcParam,model)
    #
    Highway_env_branch.sim(mpc,N_lane,1)


if __name__== "__main__":
  main()
