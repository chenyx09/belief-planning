from casadi import *
import pdb
import itertools
import numpy as np
from utils import HMM_constants, MPCParams
from scipy import sparse
from itertools import product



def X_bdry(x,bdry,width):
    dy1 = x[1]-bdry[0]-width/2
    dy2 = bdry[1]-x[1]-width/2
    if dy1<dy2:
        return dy1,np.array([0,1,0,0])
    else:
        return dy2,np.array([0,-1,0,0])

def veh_con(x,x0,umax,ignore_x=True):
    # K =    -0.3162         0   -0.8558         0
         # 0   -0.3162         0   -3.9889

    if ignore_x:
        u = np.array([-0.8558*(x[2]-x0[2]),-0.3162*(x[1]-x0[1])-3.9889*(x[3]-x0[3])])
    else:
        u = np.array([-0.3162*(x[0]-x0[0])-0.8558*(x[2]-x0[2]),-0.3162*(x[1]-x0[1])-3.9889*(x[3]-x0[3])])
    u = np.maximum(-umax,u)
    u = np.minimum(umax,u)
    return u
def dubin(x,u):
    if isinstance(x,numpy.ndarray):
        xdot = np.array([x[2]*cos(x[3]),x[2]*sin(x[3]),u[0],u[1]])
    elif isinstance(x,casadi.SX) or isinstance(x,casadi.MX):
        xdot = SX(4,1)
        xdot[0]=x[2]*cos(x[3])
        xdot[1]=x[2]*sin(x[3])
        xdot[2]=u[0]
        xdot[3]=u[1]
    return xdot
def dubin_fg(x):
    f = DM(np.array([x[2]*cos(x[3]), x[2]*sin(x[3]), 0, 0]))
    g = DM(np.array([[0, 0], [0, 0],[1, 0],[0, 1]]))
    return f,g
def dubin_f_x(x,con):
    h = 1e-6
    dudx = np.zeros([4,2])
    dudx[0] = (con(x+np.array([h,0,0,0]))-con(x-np.array([h,0,0,0])))/2/h
    dudx[1] = (con(x+np.array([0,h,0,0]))-con(x-np.array([0,h,0,0])))/2/h
    dudx[2] = (con(x+np.array([0,0,h,0]))-con(x-np.array([0,0,h,0])))/2/h
    dudx[3] = (con(x+np.array([0,0,0,h]))-con(x-np.array([0,0,0,h])))/2/h
    # pdb.set_trace()
    ja = np.concatenate((np.array([[0, 0, np.cos(x[3]), -x[2]*np.sin(x[3])],[0, 0, np.sin(x[3]), x[2]*np.cos(x[3])]])   ,dudx.transpose()))
    return ja

def generate_backup_traj(x,con,stop_crit,f0,ts=0.05,sensitivity= True):
    t = 0
    tt = []
    xx = []
    uu = []
    Q = np.identity(4)
    QQ = []

    Qt = []
    if sensitivity:
        while not stop_crit(x,t):
            QQ.append(Q)
            u = con(x)
            xdot = np.array([x[2]*np.cos(x[3]),x[2]*np.sin(x[3]),u[0],u[1]])
            ja = dubin_f_x(x,con)
            tt.append(t)
            xx.append(x)
            uu.append(u)
            Qt.append(xdot-f0)
            x = x + xdot*ts
            Q = Q + np.matmul(ja,Q)*ts
            t = t + ts
    else:
        while not stop_crit(x,t):
            u = con(x)
            xdot = np.array([x[2]*np.cos(x[3]),x[2]*np.sin(x[3]),u[0],u[1]])
            tt.append(t)
            xx.append(x)
            uu.append(u)
            x = x + xdot*ts
            t = t + ts
    return tt,xx,uu,QQ,Qt

def obs_J(x1,x2,cons):
    d = sqrt(sumsqr(x1[0:2]-x2[0:2]))-cons.R
    J = cons.J_c*softsat(-d,cons.s_c)
    return J


def softsat(x,s):
    return (np.exp(s*x)-1)/(np.exp(s*x)+1)*0.5+0.5

def backup_trans(h,cons):
    m = softsat(h,cons.s1)
    if isinstance(m,casadi.SX) or isinstance(m,casadi.MX):
        return kron((1-cons.tran_diag)*SX(np.ones([h.shape[0],1])),m.T/sum1(m))+cons.tran_diag*SX(np.eye(h.shape[0]))
    else:
        return np.kron((1-cons.tran_diag)*np.ones([h.shape[0],1]),m.T/sum(m))+cons.tran_diag*np.eye(h.shape[0])

def backup_input_prob(cbfcond,cons):
    return softsat(cbfcond-cons.c2,cons.s2)





def backup_maintain(x,cons):
    return np.array([0,-cons.Kpsi*x[3]])

def backup_brake(x,cons):
    u = np.array([softmax(vertcat(-5,-x[2]),3),-cons.Kpsi*x[3]])
    return u
def backup_lc(x,x0):
    return np.array([-0.8558*(x[2]-x0[2]),-0.3162*(x[1]-x0[1])-3.9889*(x[3]-x0[3])])
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
    if isinstance(x,numpy.ndarray):
        xs = np.empty([N,x.shape[0]])
    elif isinstance(x,casadi.SX) or isinstance(x,casadi.MX):
        xs = SX(N,x.shape[0])
    # xs = np.zeros(N,x.shape[0])
    for i in range(0,N):
        x = x+dyn(x)*ts
        xs[i,:]=x

    return xs

def lane_bdry_h(x,lb=0,ub=7.2):
    if isinstance(x,casadi.SX) or isinstance(x,casadi.MX):
        h = SX.sym('h',x.shape[0])
        for i in range(0,x.shape[0]):
            h[i] = softmin(vertcat(x[i,1]-lb,ub-x[i,1]),5)
        return h
    else:
        if x.ndim==1:
            return softmin(np.array([x[1]-lb,ub-x[1]]),5)
        else:
            h = np.zeros(x.shape[0])
            for i in range(0,x.shape[0]):
                h[i] = softmin(np.array([x[i,1]-lb,ub-x[i,1]]),5)
            return h
def veh_col(x1,x2,size,alpha=1):
    if isinstance(x1,casadi.SX) or isinstance(x1,casadi.MX):
        h = SX(x1.shape[0])
        for i in range(0,x1.shape[0]):
            dx = (fabs(x1[i,0]-x2[i,0])-size[0])/size[0]
            dy = (fabs(x1[i,1]-x2[i,1])-size[1])/size[1]
            h[i] = (dx*np.exp(alpha*dx)+dy*np.exp(dy*alpha))/(np.exp(alpha*dx)+np.exp(dy*alpha))
        return h
    else:
        if x1.ndim==1:
            dx = np.clip((abs(x1[0]-x2[0])-size[0])/size[0],-5,5)
            dy = np.clip((abs(x1[1]-x2[1])-size[1])/size[1],-5,5)

            return (dx*np.exp(alpha*dx)+dy*np.exp(dy*alpha))/(np.exp(alpha*dx)+np.exp(dy*alpha))
        else:
            h = np.zeros(x1.shape[0])
            for i in range(0,x1.shape[0]):
                dx = np.clip((abs(x1[i][0]-x2[i][0])-size[0])/size[0],-5,5)
                dy = np.clip((abs(x1[i][1]-x2[i][1])-size[1])/size[1],-5,5)
                h[i] = (dx*np.exp(alpha*dx)+dy*np.exp(dy*alpha))/(np.exp(alpha*dx)+np.exp(dy*alpha))
            return h
def obs_coll_h(x1,x2,R):

    if isinstance(x1,np.ndarray):
        if x1.ndim==1:
            h = sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)-R
        else:
            h = np.zeros(x1.shape[0])
            for i in range(0,x1.shape[0]):
                h[i] = sqrt((x1[i][0]-x2[i][0])**2+(x1[i][1]-x2[i][1])**2)-R
    elif isinstance(x1,casadi.MX) or isinstance(x1,casadi.DM) or isinstance(x1,casadi.SX):
        h = SX(x1.shape[0])
        for i in range(0,x1.shape[0]):
            h[i] = sqrt((x1[i,0]-x2[i,0])**2+(x1[i,1]-x2[i,1])**2)-R

    return h





class PredictiveModel():
    def __init__(self, n, d, N, backupcons, dt, cons,N_lane = 3):
        self.n = n # state dimension
        self.d = d # input dimention
        self.N = N # number of prediction steps
        self.m = len(backupcons)
        self.dt = dt
        self.cons = cons
        self.Asym = None
        self.Bsym = None
        self.xpsym = None
        self.hsym = None
        self.zpred = None
        self.xpred = None
        self.u0sym = None
        self.Jh = None
        self.LB = [cons.W/2,N_lane*3.6-cons.W/2]
        self.alpha = cons.alpha
        self.backupcons = backupcons
        self.calc_xp_expr()



    def dyn_linearization(self, x,u):

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
        h = SX.sym('h',x1.shape[0]*2)
        for i in range(0,x1.shape[0]):
            h[i] = veh_col(x1[i,:],x2[i,:],[self.cons.L+1,self.cons.W+0.2])
            h[i+x1.shape[0]] = lane_bdry_h(x1[i,:],self.LB[0],self.LB[1])
        return softmin(h,5)
    def branch_prob(self,h):
        m = softsat(h,self.cons.s1)
        return m/sum1(m)

    def calc_xp_expr(self):
        u = SX.sym('u', self.d)
        x = SX.sym('x',self.n)
        z = SX.sym('z',self.n)
        zdot = SX.sym('zdot',self.m,self.n)

        xp = x+dubin(x,u)*self.dt

        dyn = lambda x:dubin(x,self.backupcons[0](x))
        x1 = propagate_backup(x,dyn,self.N,self.dt)
        x2 = SX.sym('x2traj',self.N,self.n*self.m)
        for i in range(0,self.m):
            dyn = lambda x:dubin(x,self.backupcons[i](x))
            x2[:,i*self.n:(i+1)*self.n] = propagate_backup(z,dyn,self.N,self.dt)
        hi = SX.sym('h',self.m)
        for i in range(0,self.m):
            hi[i] = self.BF_traj(x2[:,self.n*i:self.n*(i+1)],x1)
        p = self.branch_prob(hi)
        # h1 = veh_col(x.T,z.T,[self.cons.L+1,self.cons.W+0.2])
        # h2 = lane_bdry_h(x.T,self.LB[0],self.LB[1])
        # h3 = lane_bdry_h(z.T,self.LB[0],self.LB[1])
        h = veh_col(x.T,z.T,[self.cons.L+1,self.cons.W+0.2],5)

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
