import numpy as np
def prox_nuclear(B,Lambda):
    """
    The proximal operator of the nuclear norm of a matrix
    min_X Lambda*||X||_*+0.5*||X-B||_F^2
    """

    [U,S,V] = np.linalg.svd(B, full_matrices=0)
    V = V.T
    svp = len(np.where(S>Lambda)[0])
    if svp>=1:
        S = S[0:svp]-Lambda
        X = np.matmul(np.matmul(U[:,0:svp], np.diag(S)), V[:,0:svp].T)
        nuclearnorm = sum(S)
    else:
        X = np.zeros(B.shape)
        nuclearnorm = 0
    return X, nuclearnorm

if __name__ == '__main__':
    a = np.array(
        [
            [1,2,3,4],
            [4,5,6,7],
            [7,8,9,10]
        ]
    )
    prox_nuclear(a,22)
