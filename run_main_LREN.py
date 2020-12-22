import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from datetime import datetime
import scipy.io as sio
import cv2
from sklearn.decomposition import PCA, KernelPCA
from lren import LREN
from calculate_falarm_rate import false_alarm_rate
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, confusion_matrix
from lrr.lrr import lrr
import time
#%matplotlib inline

def cal_BIC(log_likehood_of_model,n_components,n_samples):
    """
    claculate bayesian information criterion for selecting parameters
    """
    return n_components*np.log(n_samples)-2*np.log(log_likehood_of_model)

def create_file(file_path, msg):
    f = open(file_path,'a')
    f.write(msg)
    f.close()
    
def mkdir(path):
    import os
    
    path = path.strip()
    
    path = path.rstrip("//")
    
    isExist = os.path.exists(path)
    
    if not isExist:
        os.makedirs(path)
        print("making dir...")
        return True
    else:
        print(path+"is exist")
        return False


def parameter_setting(file_name):
    if file_name == 'HYDICE_data' or file_name == 'abu-urban-2' or file_name == 'sandiego_plane':
        clusters_num = 7
        hidden_nodes = 9
        Lam = 0.01
    elif file_name == 'abu-beach-4':
        clusters_num = 7
        hidden_nodes = 9
        Lam = 0.0001
    return clusters_num, hidden_nodes, Lam


def main():
    Folder_name = 'detection_results'
    file_name_list = ['abu-urban-2','HYDICE_data','abu-beach-4','sandiego_plane']
    
    for i in range(0,4):
                    
        time_start=time.time()

        Lambda = 100
        filenum = str(i+1)
        file_name = file_name_list[int(filenum)-1]
        print(file_name)
        clusters_num, hidden_nodes, Lam = parameter_setting(file_name)
        mkdir('./'+Folder_name+'//'+file_name+'/')
        
        
        load_fn = './datasets/avirs/'+file_name+'.mat'
        load_data = sio.loadmat(load_fn)
        load_matrix = load_data['data']
        
        load_matrix = np.array(load_matrix)
        [r, c, x_dim]=load_matrix.shape
        load_matrix = load_matrix.reshape([load_matrix.shape[0]*load_matrix.shape[1], x_dim])
        load_matrix = ((load_matrix-load_matrix.min()) /
                            (load_matrix.max()-load_matrix.min()))
        data = load_matrix

        load_fn = './datasets/avirs/'+file_name+'.mat'
        load_data = sio.loadmat(load_fn)
        anomal_target_map = load_data['map']
        anomal_target_map = np.array(anomal_target_map)

        plt.figure()
        plt.imshow(anomal_target_map)
        plt.savefig('./'+Folder_name+'//'+file_name+'/'+file_name+'target-map.png' )

        normal_data=data
        tf.reset_default_graph()

        model_lren = LREN([400,hidden_nodes], tf.nn.tanh, est_hiddens=[60,clusters_num], est_activation=tf.nn.tanh, est_dropout_ratio=0.5, epoch_size=1000, minibatch_size=int(4096)
        )

        model_lren.Perform_Density_Estimation(normal_data)
        
        Dict, S = model_lren.construct_Dict(data)

        X,E,obj,err,Iter = lrr(S.T, Dict.T, Lam)

        energy = np.linalg.norm(E.T,axis=1,ord=2)
        plt.imshow(energy.reshape(r,c))
        plt.savefig('test.png')

        energy = (energy-energy.min())/(energy.max()-energy.min())
        
        
        ret,th = cv2.threshold(np.uint8(255*energy),0,255,cv2.THRESH_OTSU)
        
        energy = energy.reshape([th.shape[0],1])

        auc_score = roc_auc_score(anomal_target_map.reshape([anomal_target_map.shape[0]*anomal_target_map.shape[1], 1]), energy)
        _, fpr_auc = false_alarm_rate(anomal_target_map.reshape([anomal_target_map.shape[0]*anomal_target_map.shape[1], 1]), energy)

        plt.figure()
        plt.imshow(energy.reshape(anomal_target_map.shape))

        plt.title('cluster centers={} hidden nodes={}\nAUC={}, FPR{},lam={}'.format(clusters_num,hidden_nodes, auc_score,fpr_auc,Lam))
        plt.savefig('./'+Folder_name+'//'+file_name+'/'+file_name+'Energy-map-cluster centers={}hidden nodes={},lam={}.png'.format(clusters_num,hidden_nodes,Lam))
        save_fn = './'+Folder_name+'//'+file_name+'/'+file_name+'Energy-map-cluster centers={}hidden nodes={},lam={}.mat'.format(clusters_num,hidden_nodes,Lam)
        save_array = energy.reshape(anomal_target_map.shape)
        sio.savemat(save_fn, {'array': save_array})
        
        
        txtfile_path = './'+Folder_name+'//'+file_name+'/'+file_name+'.txt'
        create_file(txtfile_path,file_name+'\nlambda='+str(Lambda)+'\n')
        txtfile = open('./'+Folder_name+'//'+file_name+'/'+file_name+'.txt','r+')
        txtfile.read()
        txtfile.write('\nEnergy-map cluster centers:{} hidden nodes:{}\nAUC:{}\nFPR:{}\nlam:{}\n'.format(clusters_num,hidden_nodes, auc_score,fpr_auc,Lam))

        plt.close('all')
        time_end=time.time()
        print(file_name)
        print('totally cost',time_end-time_start,'s')
        
if __name__ == '__main__':
    main()
