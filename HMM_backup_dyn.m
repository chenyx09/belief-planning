h = [-1,0,1]';
cons = HMM_con();
H = backup_trans(h,cons);

N = 20;
J = 0;

x1 = [0;0;1;0];
x2 = [2;1;1;-0.75*pi];

x = sym('x',[4,1],'real');
u = sym('x',[2,1],'real');

M = 2;
m = 2;
b = [];
for i=1:M
    b = [b;sym(['b',num2str(i)],[m,1],'real')];
end
xb = [x;b];
dyn = @dubin;
backupdyn1 = @(x)dubin(x,backup_maintain(x,cons));
backupdyn2 = @(x)dubin(x,backup_brake(x,cons));
ts = 0.1;
R = 1;
Jfun = @(x1,x2)obs_avoid(x1,x2,R,cons);
hfun = @(x1,x2)obs_coll_h(x1,x2,R);
xbackup = [];
xs=propagate_backup(x1,backupdyn1,N,ts);
xbackup = [xbackup;reshape(xs,1,[])];
xs=propagate_backup(x1,backupdyn2,N,ts);
xbackup = [xbackup;reshape(xs,1,[])];
xs=propagate_backup(x2,backupdyn1,N,ts);
xbackup = [xbackup;reshape(xs,1,[])];
xs=propagate_backup(x2,backupdyn2,N,ts);
xbackup = [xbackup;reshape(xs,1,[])];

[xbp,J] = pred(M,m,xb,dyn,u,ts,Jfun,hfun,xbackup(:,1:4),cons);
Ja = jacobian(J,u)


function [xbp,J] = pred(M,m,xb,dyn,u,ts,Jfun,hfun,xbackup,cons)
    x = xb(1:4);
    b = xb(5:end);
    b = reshape(b,m,M);
    bp = sym(0*b);
    xp = x+dyn(x,u)*ts;
    J = u'*u;
    for i=1:M
        h = sym(zeros(m,1));
        for j=1:m
            h(j)=hfun(x,xbackup(m*(i-1)+j,:)');
            J = J+b(i,j)*Jfun(x,xbackup(m*(i-1)+j,:)');
        end
        H = backup_trans(h,cons);
        bp(:,i)=H*b(:,i);
        
    end
    xbp = [xp;reshape(bp,[],1)];
    
end
function J = obs_avoid(x1,x2,R,cons)
    d = norm(x1(1:2)-x2(1:2))-R;
    J = cons.J_c*softsat(-d,cons.s_c);
end

function y = softsat(x,s)
y= (exp(s*x)-1)./(exp(s*x)+1)*0.5+0.5;
end

function H = backup_trans(h,cons)
m = softsat(h,cons.s1);
H = kron(ones(1,length(h)),m/sum(m))+cons.tran_diag*eye(length(h));
end

function p = backup_input_prob(h,dh,cons)
    p = softsat(dh+cons.alpha*h,cons.s2);
end

function [f,g] = dubin_fg(x)
f = [x(3)*cos(x(4));...
    x(3)*sin(x(4));...
    0;...
    0];
g = [0 0;...
     0 0;...
     1 0;...
     0 1];
end
function xdot = dubin(x,u)
xdot = [x(3)*cos(x(4));...
    x(3)*sin(x(4));...
    u(1);...
    u(2)];
end

function u = backup_maintain(x,cons)
u = [0;0];
end

function u = backup_brake(x,cons)
u = [max(-cons.am*x(3),-10*x(3));0];
end

function xs=propagate_backup(x,dyn,N,ts)
    xs = zeros(N,length(x));
    for i=1:N
        x = x+dyn(x)*ts;
        xs(i,:)=x';
    end
end

function h = obs_coll_h(x1,x2,R)
    if size(x1,2)==1
        x1 = x1';
    end
    if size(x2,2)==1
        x2 = x2';
    end
    h = sym(zeros(size(x1,1),1));
    for i=1:size(x1,1)
        h(i) = norm(x1(i,1:2)-x2(i,1:2))-R;
    end
end