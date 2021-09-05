from casadi import *
import pdb
import itertools
import numpy as np
from scipy import sparse
from itertools import product
from utils import Quad_constants
'''
Use CasADi to calculate the dynamics, propagate trajectories under the backup policies, and calculate the branching probabilities
'''



def quad_kinetics(x,u):
    if isinstance(x,numpy.ndarray):
        xdot = np.array([u[0]*cos(x[2])-u[1]*sin(x[2]),u[0]*sin(x[2])+u[1]*cos(x[2]),u[2]])
    elif isinstance(x,casadi.SX):
        xdot = SX(3,1)
        xdot[0]=u[0]*cos(x[2])-u[1]*sin(x[2])
        xdot[1]=u[0]*sin(x[2])+u[1]*cos(x[2])
        xdot[2]=u[2]
    elif isinstance(x,casadi.MX):
        xdot = MX(4,1)
        xdot[0]=u[0]*cos(x[2])-u[1]*sin(x[2])
        xdot[1]=u[0]*sin(x[2])+u[1]*cos(x[2])
        xdot[2]=u[2]
    return xdot


def softsat(x,s):
    return (np.exp(s*x)-1)/(np.exp(s*x)+1)*0.5+0.5


def backup_forward(x,v0):
    if isinstance(x,casadi.SX):
        u = SX(3,1)
        u[0]=v0
        return u
    elif isinstance(x,casadi.MX):
        u = SX(3,1)
        u[0]=v0
        return u
    else:
        return np.array([v0,0,0])

def backup_stop(x):
    if isinstance(x,casadi.SX):
        u = SX(3,1)
        return u
    elif isinstance(x,casadi.MX):
        u = SX(3,1)
        return u
    else:
        return np.array([0,0,0])

def softmin(x,gamma=1):
    if isinstance(x,casadi.SX) or isinstance(x,casadi.MX):
        return sum1(exp(-gamma*x)*x)/sum1(exp(-gamma*x))
    else:
        return np.sum(np.exp(-gamma*x)*x)/np.sum(np.exp(-gamma*x))

def softmax(x,gamma=1):
    if isinstance(x,casadi.SX) or isinstance(x,casadi.MX):
        return sum1(exp(gamma*x)*x)/sum1(exp(gamma*x))
    else:
        return np.sum(np.exp(gamma*x)*x)/np.sum(np.exp(gamma*x))

def propagate_backup(x,dyn,N,ts):
    '''
    Euler forward integration of the dynamics under the policy
    '''
    if isinstance(x,numpy.ndarray):
        xs = np.empty([N,x.shape[0]])
    elif isinstance(x,casadi.SX):
        xs = SX(N,x.shape[0])
    elif isinstance(x,casadi.MX):
        xs = MX(N,x.shape[0])
    # xs = np.zeros(N,x.shape[0])
    for i in range(0,N):
        x = x+dyn(x)*ts
        xs[i,:]=x
    return xs

def rot_mat(theta):
    mat = SX(2,2)
    mat[0,0]=cos(theta)
    mat[0,1]=-sin(theta)
    mat[1,0]=sin(theta)
    mat[1,1]=cos(theta)
    return mat

def robot_col1(x1,x2,L1,W1,L2,W2,tol,alpha=1):
    '''
    robot collision constraints: h>=0 means no collision
    implemented via a softmax function
    '''

    corners = np.array([[L2/2,W2/2],[L2/2,-W2/2],[-L2/2,W2/2],[-L2/2,-W2/2],[0,-W2/2],[0,W2/2]])
    if isinstance(x1,casadi.SX):
        corners = SX(corners)
        h = SX(x1.shape[0],1)
        for i in range(0,x1.shape[0]):
            theta1 = x1[i,2]
            theta2 = x2[i,2]
            delta_x0 = x2[i,0:2]-x1[i,0:2]
            T1 = rot_mat(-theta1)
            T2 = rot_mat(theta2)
            delta_x2 = (T2@corners.T).T+kron(delta_x0,SX.ones(corners.shape[0],1))
            delta_x1 = (T1@delta_x2.T).T
            h1 = SX(corners.shape[0],1)
            for j in range(0,corners.shape[0]):
                dx = fabs(delta_x1[j,0])-L1/2-tol
                dy = fabs(delta_x1[j,1])-W1/2-tol
                h1[j] = (dx*exp(alpha*dx)+dy*exp(dy*alpha))/(exp(alpha*dx)+exp(dy*alpha))
            h[i] = softmin(h1,3)
        return h
    else:
        h = np.zeros(x1.shape[0])
        for i in range(0,x1.shape[0]):
            theta1 = x1[i][2]
            theta2 = x2[i][2]
            delta_x0 = x2[i][0:2]-x1[i][0:2]
            T1 = np.array([[cos(theta1),sin(theta1)],[-sin(theta1),cos(theta1)]])
            T2 = np.array([[cos(theta2),-sin(theta2)],[sin(theta2),cos(theta2)]])
            delta_x2 = (T2@corners.T).T+delta_x0
            delta_x1 = (T1@delta_x2.T).T
            h1 = np.zeros(corners.shape[0])
            for j in range(0,corners.shape[0]):
                dx = fabs(delta_x1[j,0])-L1/2-tol
                dy = fabs(delta_x1[j,1])-W1/2-tol
                h1[j] = (dx*exp(alpha*dx)+dy*exp(dy*alpha))/(exp(alpha*dx)+exp(dy*alpha))
            h[i] = softmin(h1,3)
        return h

def robot_col(x1,x2,L1,W1,L2,W2,tol,alpha=1):
    '''
    robot collision constraints: h>=0 means no collision
    implemented via a softmax function
    '''

    if isinstance(x1,casadi.SX):
        h = SX(x1.shape[0],1)
        for i in range(0,x1.shape[0]):
            h[i] = norm_1(x1[i,0:2]-x2[i,0:2])-(L1+L2)/2-tol
        return h
    else:
        h = np.zeros(x1.shape[0])
        for i in range(0,x1.shape[0]):
            h[i] = np.linalg.norm(x1[i][0:2]-x2[i][0:2])-(L1+L2)/2-tol
        return h



class PredictiveModel():
    def __init__(self, n, d, N, backupcons, dt, cons):
        self.n = n # state dimension
        self.d = d # input dimention
        self.N = N # number of prediction steps
        self.m = len(backupcons) # number of policies
        self.dt = dt
        self.cons = cons #parameters
        self.Asym = None
        self.Bsym = None
        self.xpsym = None
        self.hsym = None
        self.zpred = None
        self.xpred = None
        self.u0sym = None
        self.Jh = None
        self.backupcons = backupcons
        self.calc_xp_expr()



    def dyn_linearization(self, x,u):
        #linearizing the dynamics x^+=Ax+Bu+C
        A = self.Asym(x,u)
        B = self.Bsym(x,u)
        xp = self.xpsym(x,u)
        C = xp-A@x-B@u

        return np.array(A),np.array(B),np.squeeze(np.array(C)),np.squeeze(np.array(xp))

    def branch_eval(self,x,z):
        p = self.psym(x,z)
        dp = self.dpsym(x,z)
        return np.array(p).flatten(),np.array(dp)
    def zpred_eval(self,z):
        return np.array(self.zpred(z))
    def xpred_eval(self,x):
        return self.xpred(x),self.u0sym(x)
    def col_eval(self,x,z):
        dh = np.squeeze(np.array(self.dhsym(x,z)))
        h = np.squeeze(np.array(self.hsym(x,z)))
        return h-np.dot(dh,x),dh

    def update_backup(self,backupcons):
        self.backupcons = backupcons
        self.m = len(backupcons)
        self.calc_xp_expr()



    def BF_traj(self,x1,x2):
        if isinstance(x1,casadi.SX):
            h = SX(x1.shape[0],1)
        elif isinstance(x1,casadi.MX):
            h = MX(x1.shape[0],1)
        for i in range(0,x1.shape[0]):
            h[i] = robot_col(x1[i,:],x2[i,:],self.cons.L1,self.cons.W1,self.cons.L2,self.cons.W2,self.cons.col_tol)
        return softmin(h,5)
    def branch_prob(self,h):
        #branching probablity as a function of the safety function
        # m = softsat(h,self.cons.s1)
        m = exp(self.cons.s1*h)
        return m/sum1(m)

    def calc_xp_expr(self):
        u = SX.sym('u', self.d)
        x = SX.sym('x',self.n)
        z = SX.sym('z',self.n)
        zdot = SX.sym('zdot',self.m,self.n)

        xp = x+quad_kinetics(x,u)*self.dt

        dyn = lambda x:quad_kinetics(x,self.backupcons[0](x))
        x1 = propagate_backup(x,dyn,self.N,self.dt)
        x2 = SX(self.N,self.n*self.m)
        for i in range(0,self.m):
            dyn = lambda x:quad_kinetics(x,self.backupcons[i](x))
            x2[:,i*self.n:(i+1)*self.n] = propagate_backup(z,dyn,self.N,self.dt)
        hi = SX(self.m,1)
        for i in range(0,self.m):
            hi[i] = self.BF_traj(x2[:,self.n*i:self.n*(i+1)],x1)
        p = self.branch_prob(hi)

        h = robot_col(x.T,z.T,self.cons.L1,self.cons.W1,self.cons.L2,self.cons.W2,self.cons.col_tol,1)

        self.xpsym = Function('xp',[x,u],[xp])
        self.Asym = Function('A',[x,u],[jacobian(xp,x)])
        self.Bsym = Function('B',[x,u],[jacobian(xp,u)])
        self.dhsym = Function('dh',[x,z],[jacobian(h,x)])
        self.hsym = Function('h',[x,z],[h])
        self.zpred = Function('zpred',[z],[x2])
        self.xpred = Function('xpred',[x],[x1])
        self.psym = Function('p',[x,z],[p])
        self.dpsym = Function('dp',[x,z],[jacobian(p,x)])
        self.u0sym = Function('u0',[x],[self.backupcons[0](x)])

# x1 = SX([[1,1,0.2],[1,1,0.2]])
# x2 = SX([[4,3,-0.2],[4,3,-0.2]])
# x1 = np.array([1,1,0.2])
# x2 = np.array([4,3,-0.2])
# L1 = 3
# L2 = 2
# W1 = 2
# W2 = 1.5
# vxm = 1
# vym = 0.2
# rm = 0.5
# dt = 0.05
# v0 = 1
# n = 3
# d = 3
# N =10
# u = np.array([0.5,0.1,0.2])
# backupcons = [lambda x:backup_forward(x,v0),lambda x:backup_stop(x)]
# cons = Quad_constants(s1=2,s2=3,c2=0.5,alpha=1,R=1.2,vxm=vxm,vym=vym,rm=rm,L1=L1, W1=W1,L2=L2,W2=W2,col_alpha=5)
# model = PredictiveModel(n, d, N, backupcons, dt, cons)
# h,dh = model.col_eval(x1,x2)
# p,dp = model.branch_eval(x1,x2)
# pdb.set_trace()
