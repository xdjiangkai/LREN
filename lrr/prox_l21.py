import numpy as np
def prox_l21(B,Lambda):
    """
    The proximal operator of the l21 norm of a matrix
    l21 norm is the sum of the l2 norm of all columns of a matrix 
    min_X lambda*||X||_{2,1}+0.5*||X-B||_2^2
    """
    X = np.zeros(B.shape)
    for i in range(X.shape[1]):
        nxi = np.linalg.norm(B[:,i])
        if nxi > Lambda:
            X[:,i] = (1-Lambda/nxi)*B[:,i]
        
    return X