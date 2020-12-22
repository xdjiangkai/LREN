import numpy as np
def prox_l1(b,Lambda):
    """
    The proximal operator of the l1 norm
    min_x lambda*||x||_1+0.5*||x-b||_2^2
    """
    # x = np.max([0,b-Lambda])+np.min([0,b+Lambda])
    x = np.where(b-Lambda>0, b-Lambda, 0) + np.where(b+Lambda<0, b+Lambda, 0)
    return x