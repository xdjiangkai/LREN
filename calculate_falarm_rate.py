import numpy as np

def false_alarm_rate(target, predicted):
    """
    calculate AUC and false alarm auc
    
    Input:
        target.shape = [n,1]
        predicted.shape = [n,1]
    
    Output:
        PD_PF_auc
        PF_tau_auc
    """
    target = ((target-target.min()) /
                            (target.max()-target.min()))
    predicted = ((predicted-predicted.min()) /
                            (predicted.max()-predicted.min()))
    anomaly_map = target
    normal_map = 1-target
    num = 30000
    taus = np.linspace(0,predicted.max(),num=num)
    PF = np.zeros([num,1])
    PD = np.zeros([num,1])
    for index in range(num):
        tau = taus[index]
        anomaly_map_1 = np.double(predicted>=tau)
        PF[index] = np.sum(anomaly_map_1*normal_map)/np.sum(normal_map)
        PD[index] = np.sum(anomaly_map_1*anomaly_map)/np.sum(anomaly_map)
    """
    plt.figure()
    plt.plot(PF,PD)
    plt.figure()
    plt.plot(taus,PD)
    plt.figure()
    plt.plot(taus,PF)
    plt.show()
    """
    PD_PF_auc = np.sum((PF[0:num-1,:]-PF[1:num,:])*(PD[1:num]+PD[0:num-1])/2)
    #area0 = np.trapz(PD.squeeze(),PF.squeeze())
    PF_tau_auc = np.trapz(PF.squeeze(),taus.squeeze())
    return PD_PF_auc, PF_tau_auc
