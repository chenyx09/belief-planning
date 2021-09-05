import pdb
import osqp
import argparse
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
import numpy as np
from scipy.io import loadmat
from scipy import interpolate, sparse
import random
import math
from numpy import linalg as LA
from numpy.linalg import norm
from quadruped_branch_dyn import *



def with_probability(P=1):
    return np.random.uniform() <= P

class robot():
    def __init__(self, state=[0,0,0],L=1,W=0.5,dt=0.05,backupidx=0):
        self.state = np.array(state)
        self.dt = dt
        self.L = L
        self.W = W
        self.x_pred = []
        self.y_pred = []
        self.xbackup = None
        self.backupidx = backupidx
    def step(self,u): # controlled robot
        dxdt = np.array([u[0]*np.cos(self.state[2])-u[1]*np.sin(self.state[2]),
                         u[1]*np.cos(self.state[2])+u[0]*np.sin(self.state[2]),
                         u[2]])
        self.state = self.state + dxdt*self.dt


class Quad_env():
    def __init__(self,NR,mpc,x_des):
        '''
        Input: NR: number of vehicles
               mpc: mpc controller for the controlled robot
        '''
        self.dt = mpc.predictiveModel.dt
        self.robot_set = []
        self.NR = NR
        self.desired_x = [None]*NR
        self.mpc = mpc
        self.predictiveModel = mpc.predictiveModel
        self.backupcons = mpc.predictiveModel.backupcons

        self.m = len(self.backupcons)
        self.cons = mpc.predictiveModel.cons

        x0 = np.array([[0,1.8,0],[2.5,2.5,-np.pi/2]])
        self.robot_set.append(robot(x0[0],L=self.cons.L1,W=self.cons.W1, dt=self.dt,backupidx = 0))
        self.desired_x[0] = x_des
        for i in range(1,self.NR):
            self.robot_set.append(robot(x0[i],L=self.cons.L2,W=self.cons.W2, dt=self.dt,backupidx = 0))
            self.desired_x[i] = x0[i]



    def step(self,t_):
        # initialize the trajectories to be propagated forward under the backup policy
        u_set  = [None]*self.NR
        xx_set = [None]*self.NR
        u0_set = [None]*self.NR
        x_set  = [None]*self.NR

        umax = np.array([self.cons.vxm,self.cons.vym,self.cons.rm])
        # generate backup trajectories
        self.xbackup = np.empty([0,(self.mpc.N+1)*3])
        for i in range(0,self.NR):
            z = self.robot_set[i].state
            xx_set[i] = self.predictiveModel.zpred_eval(z)


        idx0 = self.robot_set[0].backupidx
        n = self.predictiveModel.n
        x1 = xx_set[0][:,idx0*n:(idx0+1)*n]
        for i in range(0,self.NR):
            if i!=0:
                hi = np.zeros(self.m)
                for j in range(0,self.m):
                    hi[j] = min(robot_col(x1,xx_set[i][:,j*n:(j+1)*n],self.robot_set[0].L,self.robot_set[0].W,self.robot_set[i].L,self.robot_set[i].W,self.cons.col_tol))

                if hi[0]>0.5:
                    self.robot_set[i].backupidx=0;
                else:
                    self.robot_set[i].backupidx = np.argmax(hi)

            u0_set[i]=self.backupcons[self.robot_set[i].backupidx](self.robot_set[i].state)


        # set x_ref for the overtaking maneuver and call the MPC
        dx = self.desired_x[0][0:2]-self.robot_set[0].state[0:2]
        dx = dx/norm(dx)*min(norm(dx),5.0)

        if norm(dx)>0.1:
            psiRef = arctan2(dx[1],dx[0])
            while psiRef-self.desired_x[0][2]>np.pi:
                psiRef-=2*np.pi
            while psiRef-self.desired_x[0][2]<-np.pi:
                psiRef+=2*np.pi
        else:

            psiRef = self.robot_set[0].state[2]
        xRef = self.robot_set[0].state.copy()
        xRef[0:2]+=dx
        xRef[2] = psiRef
        print(xRef)

        self.mpc.solve(self.robot_set[0].state,self.robot_set[1].state,xRef)
        # pdb.set_trace()
        u_set[0] = self.mpc.uPred[0]
        xPred,zPred,uPred = self.mpc.BT2array()
        self.robot_set[0].step(u_set[0])
        x_set[0] = self.robot_set[0].state
        # print(psiRef)

        for i in range(1,self.NR):
            u_set[i] = u0_set[i]
            self.robot_set[i].step(u_set[i])
            x_set[i] = self.robot_set[i].state

        return u_set,x_set,xx_set,xPred,zPred


def Robot_sim(env,T):
    # simulate the scenario
    collision = False
    dt = env.dt
    t=0
    N = int(round(T/dt))
    state_rec = np.zeros([env.NR,N,3])
    b_rec = [None]*N
    backup_rec = [None]*env.NR
    backup_choice_rec = [None]*env.NR
    xPred_rec = [None]*N
    zPred_rec = [None]*N
    for i in range(0,env.NR):
        backup_rec[i]=[None]*N
        backup_choice_rec[i] = [None]*N
    input_rec = np.zeros([env.NR,N,3])
    for i in range(0,len(env.robot_set)):
        state_rec[i][t]=env.robot_set[i].state

    xx_set = []
    while t<N:
        print("t=",t*env.dt)
        u_set,x_set,xx_set,xPred,zPred=env.step(t)
        xPred_rec[t]=xPred
        zPred_rec[t]=zPred
        for i in range(0,env.NR):
            input_rec[i][t]=u_set[i]
            state_rec[i][t]=x_set[i]
            backup_rec[i][t]=xx_set[i]
            backup_choice_rec[i][t] = env.robot_set[i].backupidx
        t=t+1
    return state_rec,input_rec,backup_rec,backup_choice_rec,xPred_rec,zPred_rec

def plot_snapshot(x,z,env,idx=None,varycolor=True,zpatch=True,arrow = True):
    '''
    Ploting a snapshot of the simulation, for debugging
    '''

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ego_idx = 0
    ego_robot = env.robot_set[ego_idx]
    robot_patch = [None]*env.NR
    for i in range(0,env.NR):
        rob = env.robot_set[i]
        if i==ego_idx:
            robot_patch[i]=plt.Rectangle((rob.state[0]-rob.L/2,rob.state[1]-rob.W/2), rob.L,rob.W, fc='r', zorder=0)
        else:
            robot_patch[i]=plt.Rectangle((rob.state[0]-rob.L/2,rob.state[1]-rob.W/2), rob.L,rob.W, fc='b', zorder=0)

    ego_y = ego_robot.state[1]
    ego_x = ego_robot.state[0]

    xmin = ego_x-25
    xmax = ego_x+25
    ymin = ego_y-25
    ymax = ego_y+25
    try:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(-ymax,-ymin )
    except:
        pdb.set_trace()
    ts = ax.transData
    for i in range(0,env.NR):
        coords = ts.transform([env.robot_set[i].state[0],-env.robot_set[i].state[1]])
        tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], env.robot_set[i].state[2])
        te= ts + tr
        robot_patch[i].set_xy([env.robot_set[i].state[0]-env.robot_set[i].L/2,env.robot_set[i].state[1]-env.robot_set[i].W/2])
        robot_patch[i].set_transform(te)
        ax.add_patch(robot_patch[i])

    colorset = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','y','m','c','g']
    xPred,zPred,uPred = env.mpc.BT2array()
    if idx is None:
        idx = range(0,len(zPred))

    for j in idx:
        for k in range(0,xPred[j].shape[0]):
            if k%2==1:
                if varycolor:
                    x_patch = plt.Rectangle((xPred[j][k,0]-env.robot_set[ego_idx].L/2,xPred[j][k,1]-env.robot_set[ego_idx].W/2), env.robot_set[ego_idx].L,env.robot_set[ego_idx].W,ec=colorset[j], fc=colorset[j],alpha=0.2, zorder=0)
                else:
                    x_patch = plt.Rectangle((xPred[j][k,0]-env.robot_set[ego_idx].L/2,xPred[j][k,1]-env.robot_set[ego_idx].W/2), env.robot_set[ego_idx].L,env.robot_set[ego_idx].W,ec='y', fc='y',alpha=0.2, zorder=0)
                coords = ts.transform([xPred[j][k,0],-xPred[j][k,1]])
                # if arrow:
                #     arr = ax.arrow(xPred[j][k,0],-xPred[j][k,1],uPred[j][k,0]*np.cos(xPred[j][k,3]),-uPred[j][k,0]*np.sin(xPred[j][k,3]),head_width=0.5,length_includes_head=True)
                # else:
                #     arr = None
                tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], -xPred[j][k,2])
                x_patch.set_transform(ts+tr)
                ax.add_patch(x_patch)

    for j in idx:
        z_patch = plt.plot(zPred[j][:,0],-zPred[j][:,1],'m--',linewidth = 2)[0]
        if zpatch:
            for k in range(0,zPred[j].shape[0]):
                if k%2==1:
                    if varycolor:
                        z_patch = plt.Rectangle((zPred[j][k,0]-env.robot_set[1].L/2,zPred[j][k,1]-env.robot_set[1].W/2), env.robot_set[1].L,env.robot_set[1].W,ec=colorset[-1-j], fc=colorset[-1-j],alpha=0.2, zorder=0)
                    else:
                        z_patch = plt.Rectangle((zPred[j][k,0]-env.robot_set[1].L/2,zPred[j][k,1]-env.robot_set[1].W/2), env.robot_set[1].L,env.robot_set[1].W,ec='c', fc='c',alpha=0.2, zorder=0)
                    coords = ts.transform([zPred[j][k,0],-zPred[j][k,1]])
                    tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], -zPred[j][k,2])
                    z_patch.set_transform(ts+tr)
                    ax.add_patch(z_patch)

    ax.legend([x_patch,z_patch,arr],['Planned trajectory for ego robot','Predicted trajectory for the uncontrolled robot','Ego robot acceleration'],fontsize=15)
    plt.show()


def animate_scenario(env,state_rec,backup_rec,backup_choice_rec,xPred_rec,zPred_rec,x_des,output=None):
    '''
    Animate the simulation
    '''
    if output:
        matplotlib.use("Agg")
    ego_idx = 0
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    plt.grid()

    nframe = len(state_rec[0])
    ego_veh = env.robot_set[ego_idx]
    robot_patch = [None]*env.NR
    for i in range(0,env.NR):
        rob = env.robot_set[i]
        if i==ego_idx:
            robot_patch[i]=plt.Rectangle((rob.state[0]-rob.L/2,rob.state[1]-rob.W/2), rob.L,rob.W, fc='r', zorder=0)
        else:
            robot_patch[i]=plt.Rectangle((rob.state[0]-rob.L/2,rob.state[1]-rob.W/2), rob.L,rob.W, fc='b', zorder=0)
    for patch in robot_patch:
        ax.add_patch(patch)


    def animate(t,robot_patch,state_rec,backup_rec,backup_choice_rec,xPred_rec,zPred_rec,env,x_des,ego_idx=0):
        N_robot = len(state_rec)
        ego_y = state_rec[ego_idx][t][1]
        ego_x = state_rec[ego_idx][t][0]
        ax.clear()
        ax.grid()
        xmin = -10
        xmax = 10
        ymin = -10
        ymax = 10
        dest_patch = patches.Circle((x_des[0],x_des[1]),radius=0.3,fill=False,ec='c')
        ax.add_patch(dest_patch)
        try:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(-ymax,-ymin )
        except:
            pdb.set_trace()
        ts = ax.transData
        for i in range(0,N_robot):
            coords = ts.transform([state_rec[i][t][0],state_rec[i][t][1]])
            tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], state_rec[i][t][2])
            te= ts + tr
            robot_patch[i].set_xy([state_rec[i][t][0]-env.robot_set[i].L/2,state_rec[i][t][1]-env.robot_set[i].W/2])
            robot_patch[i].set_transform(te)
            ax.add_patch(robot_patch[i])
            idx = backup_choice_rec[i][t]


        colorset = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan','y','m','c','g']
        for j in range(0,len(xPred_rec[t])):
            plt.plot(xPred_rec[t][j][:,0],xPred_rec[t][j][:,1],'b--',linewidth = 1)
            for k in range(0,xPred_rec[t][j].shape[0]):
                if k%4==1:
                    newpatch = plt.Rectangle((xPred_rec[t][j][k,0]-env.robot_set[ego_idx].L/2,xPred_rec[t][j][k,1]-env.robot_set[ego_idx].W/2), env.robot_set[ego_idx].L,env.robot_set[ego_idx].W, fc=colorset[j],alpha=0.2, zorder=0)
                    coords = ts.transform([xPred_rec[t][j][k,0],xPred_rec[t][j][k,1]])
                    tr = matplotlib.transforms.Affine2D().rotate_around(coords[0], coords[1], xPred_rec[t][j][k,2])
                    newpatch.set_transform(ts+tr)
                    ax.add_patch(newpatch)

        for j in range(0,len(zPred_rec[t])):
            plt.plot(zPred_rec[t][j][:,0],zPred_rec[t][j][:,1],'r--',linewidth = 1)

        return robot_patch
    anim = animation.FuncAnimation(fig, animate, fargs=(robot_patch,state_rec,backup_rec,backup_choice_rec,xPred_rec,zPred_rec,env,x_des,ego_idx,),
                                   frames=nframe,
                                   interval=env.dt*1000,
                                   blit=False, repeat=False)


    if output:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=int(1/env.dt), metadata=dict(artist='Me'), bitrate=1800)
        anim_name = output
        anim.save(anim_name,writer=writer)
    else:
        plt.show()



def sim(mpc):
    x_des = np.array([5.,-3.,0.])
    env = Quad_env(NR=2,mpc = mpc, x_des = x_des)
    state_rec,input_rec,backup_rec,backup_choice_rec,xPred_rec,zPred_rec = Robot_sim(env,40)
    pdb.set_trace()
    animate_scenario(env,state_rec,backup_rec,backup_choice_rec,xPred_rec,zPred_rec,x_des)
