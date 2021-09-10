import numpy as np
import scipy.io as sio
from .prox_nuclear import prox_nuclear
from .prox_l1 import prox_l1
from .prox_l21 import prox_l21
def lrr(A,B,Lambda,opts=None):
    """
     M-ADMM
     min_{X,E} ||X||_*+lambda*loss(E), s.t. A=BX+E
     loss(E) = ||E||_1 or 0.5*||E||_F^2 or ||E||_{2,1} 
     #############################################################
     Input:
           A       -    d*na matrix
           B       -    d*nb matrix
           lambda  -    >0, parameter
           opts    -    Structure value in Matlab. The fields are
               opts.loss       -   'l1': loss(E) = ||E||_1 
                                   'l2': loss(E) = 0.5*||E||_F^2
                                   'l21' (default): loss(E) = ||E||_{2,1}
               opts.tol        -   termination tolerance
               opts.max_iter   -   maximum number of iterations
               opts.mu         -   stepsize for dual variable updating in ADMM
               opts.max_mu     -   maximum stepsize
               opts.rho        -   rho>=1, ratio used to increase mu
               opts.DEBUG      -   0 or 1
     ############################################################
     Output:
           X       -    nb*na matrix
           E       -    d*na matrix
           obj     -    objective function value
           err     -    residual
           iter    -    number of iterations
    """
    tol = 1e-8
    max_iter = 500
    rho = 1.2###############
    mu = 1e-3##########
    max_mu = 1e10
    DEBUG = 0
    loss = 'l21'

    d,na = A.shape
    _,nb = B.shape

    X = np.zeros([nb,na])
    E = np.zeros([d,na])
    J = X

    Y1 = E
    Y2 = X
    BtB = np.matmul(B.T,B)
    BtA = np.matmul(B.T,A)
    I = np.eye(nb)
    invBtBI = np.matmul(np.linalg.inv(BtB+I),I)#invBtBI = (BtB+I)\I;

    Iter = 0
    for Iter in range(max_iter):
        Xk = X
        Ek = E
        Jk = J
        # first super block {J,E}
        [J,nuclearnormJ] = prox_nuclear(X+Y2/mu,1/mu)

        if loss == "l1":
            E = prox_l1(A-np.matmul(B,X)+Y1/mu,Lambda/mu)
        elif loss == "l21":
            E = prox_l21(A-np.matmul(B,X)+Y1/mu,Lambda/mu)
        elif loss == "l2":
            E = mu*(A-np.matmul(B,X)+Y1/mu)/(Lambda+mu)
        else:
            print('not supported loss function')

        # second  super block {X}
        # X = invBtBI*(B'*(Y1/mu-E)+BtA-Y2/mu+J);
        temp = (np.dot(B.T,(Y1/mu-E))+BtA-Y2/mu+J)
        X = np.dot(invBtBI,temp)
    
        dY1 = A-np.matmul(B, X)-E
        dY2 = X-J
        chgX = np.max(np.abs(Xk-X))
        chgE = np.max(np.abs(Ek-E))
        chgJ = np.max(np.abs(Jk-J))
        chg = np.max(np.array([chgX,chgE,chgJ,np.max(np.abs(dY1)),np.max(np.abs(dY2))]))
        if DEBUG : 
            if Iter == 1 or mod(Iter, 10) == 0 : 
                obj = nuclearnormJ+Lambda*comp_loss(E,loss)
                err = sqrt(norm(dY1,'fro')^2+norm(dY2,'fro')^2)
                print('iter '+str(Iter)+', mu='+str(mu)+', obj='+str(obj)+', err='+str(err)) 

        
        if chg < tol:
            break

        Y1 = Y1 + np.dot(mu, dY1)
        Y2 = Y2 + np.dot(mu, dY2)
        mu = np.min([np.dot(rho,mu),max_mu])

    obj = nuclearnormJ+Lambda*comp_loss(E,loss)
    err = np.sqrt(np.power(np.linalg.norm(dY1,ord='fro'),2)+np.power(np.linalg.norm(dY2,ord='fro'),2))
    return X,E,obj,err,Iter
def comp_loss(E,loss): 

    if loss == 'l1':
        out = np.linalg.norm(E,ord=1)
    elif loss == 'l21': 
        out = 0
        for i in range(E.shape[1]):
            out = out + np.linalg.norm(E[:,i])
    elif loss == 'l2':
        out = 0.5*np.power(np.linalg.norm(E,'fro'),2)
    return out

if __name__ == '__main__':
    load_fn = './A.mat'  # 100*100*205
    load_data = sio.loadmat(load_fn)
    A = load_data['A']
    Lambda = 0.001
    lrr(A,A,Lambda)