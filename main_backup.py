import sys
# sys.path.append('fnc/simulator')
# sys.path.append('fnc/controller')
# sys.path.append('fnc')
import matplotlib.pyplot as plt
from Init_MPC import initMPCParams
from PredictiveControllers import MPC, MPCParams
from HMM_backup_dyn import *
import Highway_env
import numpy as np
import pickle
import pdb

def main():
    # ======================================================================================================================
    # ============================================= Initialize parameters  =================================================
    # ======================================================================================================================
    N = 15                                    # Horizon length
    nx = 4;   d = 2                            # State and Input dimension
    M = 3
    m = 2
    n = nx+M*m
    x0 = np.array([0, 1.8, 0, 0])       # Initial condition
    am = 6.0
    rm = 0.3
    dt = 0.1

    N_lane = 4

    # Initialize controller parameters

    cons =  HMM_constants(s1=2,s2=3,c2=0.5,tran_diag=0.3,alpha=1,R=1.2,am=am,rm=rm,J_c=20,s_c=1,ylb = 0.,yub = 7.2,L=4, W=2.5,col_alpha=5,Kpsi = 0.5)
    backupcons = [lambda x:backup_maintain(x,cons),lambda x:backup_brake(x,cons)]
    model = PredictiveModel(nx,d,M,backupcons,dt,cons)
    xuc = np.array([[0,0,1,0],[6,3,1,0],[-5,3,1,0.1]])
    x = np.array([0.5,0.6,1,-0.1])
    b = 1.0/m*np.ones([M,m])
    xb = vertcat(x,reshape(b,-1,1))
    xRef = np.array([0.5,1.8,15,0])
    mpcParam = initMPCParams(nx, d, N,M,m, 1.8,20,am,rm,N_lane,cons.W)

    # # Init simulators
    # simulator     = Simulator(map)

    # ======================================================================================================================
    # ======================================= PID path following ===========================================================
    # ======================================================================================================================


    # ======================================================================================================================
    # ======================================  LINEAR REGRESSION ============================================================
    # ======================================================================================================================
    print("Starting MPC")

    # Initialize MPC and run closed-loop sim
    mpc = MPC(mpcParam,model)

    Highway_env.sim(mpc,N_lane)
    # xMPC_cl, uMPC_cl, xMPC_cl_glob, _ = simulator.sim(xS, mpc)
    print("===== MPC terminated")

    # xbackup = model.generate_backup_traj(xuc,mpc.N)
    # mpc.solve(x,b,xbackup,xRef)
    # pdb.set_trace()
    # animation_states(map, LMPCOpenLoopData, lmpc, Laps-2)
    # animation_states(map, LMPCOpenLoopData, lmpc, Laps-2)
    # animation_states(map, LMPCOpenLoopData, lmpc, Laps-2)
    # animation_states(map, LMPCOpenLoopData, lmpc, Laps-2)
    # saveGif_xyResults(map, LMPCOpenLoopData, lmpc, Laps-2)

if __name__== "__main__":
  main()
