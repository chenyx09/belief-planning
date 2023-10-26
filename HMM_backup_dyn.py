from casadi import *
import pdb
import itertools
import numpy as np
from utils import HMM_constants, MPCParams
from scipy import sparse



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
    u = np.array([softmax(-5,-x[2],3),-cons.Kpsi*x[3]])
    return u
def softmin(x,y,gamma=1):
    return (exp(-gamma*x)*x+exp(-gamma*y)*y)/(exp(-gamma*x)+exp(-gamma*y))

def softmax(x,y,gamma=1):
    return (exp(gamma*x)*x+exp(gamma*y)*y)/(exp(gamma*x)+exp(gamma*y))

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
    return softmin(x[1]-lb,ub-x[1],5)
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
    def __init__(self, n, d, M,backupcons,dt, cons):
        self.n = n # state dimension
        self.d = d # input dimention
        self.M = M # number of uncontrolled agents
        self.m = len(backupcons)
        self.lamb = 0.0 # regularization
        self.dt = dt
        self.cons = cons
        self.Asym = None
        self.Bsym = None
        self.xbpsym = None
        self.Jsym = None
        self.hsym = None
        self.Jh = None
        self.hsym = None
        self.Hcostx = None
        self.Jcostx = None
        self.Hcostu = None
        self.Jcostu = None
        self.alpha = cons.alpha
        self.backupcons = backupcons
        self.calc_xp_expr()




    def generate_backup_traj(self,x0,N):
        if isinstance(x0,numpy.ndarray):
            xbackup = np.empty([self.M*self.m,N*self.n])
        elif isinstance(x0,casadi.SX) or isinstance(x0,casadi.MX):
            xbackup = SX(self.M*self.m,N*self.n)
        for i in range(0,self.M):
            for j in range(0,self.m):
                dyn = lambda x:dubin(x,self.backupcons[j](x))
                xs = propagate_backup(x0[i],dyn,N,self.dt)
                xbackup[self.m*i+j,:]=reshape(xs,1,-1)
        return xbackup

    def regressionAndLinearization(self, xb,xbackup, u):

        A = self.Asym(xb,u,xbackup)
        B = self.Bsym(xb,u,xbackup)
        xbp = self.xbpsym(xb,u,xbackup)
        h0 = [None]*self.M
        Jh = [None]*self.M
        for i in range(0,self.M):
            Jhi = self.Jh[i](xb,xbackup)
            h0i = self.hsym[i](xb,xbackup)-Jhi@xb
            Jh[i] = np.array(Jhi)
            h0[i] = np.array(h0i)


        # Q = self.Hcostx(xb,u,xbackup)
        # fx = self.Jcostx(xb,u,xbackup)
        # R = self.Hcostu(xb,u,xbackup)
        # fu = self.Jcostu(xb,u,xbackup)

        C = xbp-A@xb-B@u

        return np.array(A),np.array(B),np.array(np.squeeze(C)),h0,Jh
    def calc_xp_expr(self):
        u = SX.sym('u', self.d)
        b = SX.sym('b',self.M,self.m)
        x = SX.sym('x',self.n)
        xbackup = SX.sym('xbackup',self.M*self.m,self.n)
        xp = x+dubin(x,u)*self.dt
        xb = vertcat(x,reshape(b,-1,1))
        J = sumsqr(u)
        self.hsym = [None]*self.M
        self.Jh = [None]*self.M
        bp = SX(self.M,self.m)
        # hsum = SX(self.M,1)
        for i in range(0,self.M):
            h = SX(self.m,1)
            for j in range(0,self.m):

                # h[j]=softmin(obs_coll_h(x.T,xbackup[self.m*i+j,:],self.cons.R),lane_bdry_h(xbackup[self.m*i+j,:],self.cons.ylb,self.cons.yub),5)
                h[j]=softmin(veh_col(x.T,xbackup[self.m*i+j,:],[self.cons.L+1,self.cons.W+0.2]),lane_bdry_h(xbackup[self.m*i+j,:],self.cons.ylb,self.cons.yub),self.cons.col_alpha)
                # J = J+b[i,j]*obs_J(x.T,xbackup[self.m*i+j,:],cons)
            H = backup_trans(h,self.cons)
            bp[i,:]=b[i,:]@H
            self.hsym[i] = Function('h0',[xb,xbackup],[h])
            self.Jh[i] = Function('Jh',[xb,xbackup],[jacobian(h,xb)])

            # hsum[i] = b[i,:]@h

        xbp = vertcat(xp,reshape(bp,-1,1))
        self.xbpsym = Function('xbp',[xb,u,xbackup],[xbp])
        self.Asym = Function('A',[xb,u,xbackup],[jacobian(xbp,xb)])
        self.Bsym = Function('B',[xb,u,xbackup],[jacobian(xbp,u)])
        # self.Jsym = Function('J',[xb,u,xbackup],[J])
        # self.hsym = Function('Jh',[xb,xbackup],[hsum])
        # self.Jh = Function('Jh',[xb,xbackup],[jacobian(hsum,xb)])
        # [Hx,fx] = hessian(J,xb)
        # [Hu,fu] = hessian(J,u)
        # self.Hcostx = Function('Hcostx',[xb,u,xbackup],[Hx])
        # self.Hcostu = Function('Hcostu',[xb,u,xbackup],[Hu])
        # self.Jcostx = Function('Jcostx',[xb,u,xbackup],[fx])
        # self.Jcostu = Function('Jcostu',[xb,u,xbackup],[fu])

    def compute_Q_M(self, inputFeatures, usedIt):
        Counter = 0
        X0   = np.empty((0,len(self.stateFeatures)+len(inputFeatures)))
        Ktot = np.empty((0))

        for it in usedIt:
            X0 = np.append( X0, np.hstack((self.xStored[it][np.ix_(self.indexSelected[Counter], self.stateFeatures)],self.uStored[it][np.ix_(self.indexSelected[Counter], inputFeatures)])), axis=0 )
            Ktot    = np.append(Ktot, self.K[Counter])
            Counter += 1

        M = np.hstack( (X0, np.ones((X0.shape[0], 1))) )
        Q0 = np.dot(np.dot(M.T, np.diag(Ktot)), M)
        Q = matrix(Q0 + self.lamb * np.eye(Q0.shape[0]))

        return Q, M

    def compute_b(self, yIndex, usedIt, M):
        Counter = 0
        y = np.empty((0))
        Ktot = np.empty((0))

        for it in usedIt:
            y       = np.append(y, np.squeeze(self.xStored[it][self.indexSelected[Counter] + 1, yIndex]))
            Ktot    = np.append(Ktot, self.K[Counter])
            Counter += 1

        b = matrix(-np.dot(np.dot(M.T, np.diag(Ktot)), y))
        return b
