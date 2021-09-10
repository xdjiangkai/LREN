import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
#import joblib

from lren.SpectralMappingNet import SpectralMappingNet
from lren.DensityEstimationNet import DensityEstimationNet
from lren.GaussianMixtureModel import GaussianMixtureModel

class LREN:
    def __init__(self, ae_hiddens, ae_activation,
            est_hiddens, est_activation, est_dropout_ratio=0.5,
            minibatch_size=1024, epoch_size=100,
            learning_rate=0.0001, lambda1=0.1, lambda2=0.0001,
            normalize=True, random_seed=123):

        self.comp_net = SpectralMappingNet(ae_hiddens, ae_activation)
        self.est_net = DensityEstimationNet(est_hiddens, est_activation)
        self.est_dropout_ratio = est_dropout_ratio

        n_comp = est_hiddens[-1]
        self.gmm = GaussianMixtureModel(n_comp)

        self.minibatch_size = minibatch_size
        self.epoch_size = epoch_size
        self.learning_rate = learning_rate
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.normalize = normalize
        self.scaler = None
        self.seed = random_seed

        self.graph = None
        self.sess = None

    def __del__(self):
        if self.sess is not None:
            self.sess.close()

    def Perform_Density_Estimation(self, x):
        """ 
        Perform density estimation
        """
        n_samples, n_features = x.shape

        if self.normalize:
            self.scaler = scaler = StandardScaler()
            x = scaler.fit_transform(x)

        with tf.Graph().as_default() as graph:
            self.graph = graph
            tf.set_random_seed(self.seed)
            np.random.seed(seed=self.seed)

            self.input = input = tf.placeholder(
                dtype=tf.float32, shape=[None, n_features])
            self.drop = drop = tf.placeholder(dtype=tf.float32, shape=[])

            z, x_dash  = self.comp_net.inference(input)
            
            self.z = z
            
            gamma = self.est_net.inference(z, drop)
            self.gamma = gamma

            self.gmm.Gaussian_Mixture_Model_Parameter_Estimation(z, gamma)
            energy = self.gmm.Calculate_Energy(z)

            self.x_dash = x_dash

            loss = (self.comp_net.reconstruction_error(input, x_dash) +
                self.lambda1 * tf.reduce_mean(energy) +
                self.lambda2 * self.gmm.Cov_Diag_Loss())
            rec_loss = self.comp_net.reconstruction_error(input, x_dash)
            minimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
            rec_minimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(rec_loss)
            n_batch = (n_samples - 1) // self.minibatch_size + 1

            init = tf.global_variables_initializer()

            self.sess = tf.Session(graph=graph)
            self.sess.run(init)

            idx = np.arange(x.shape[0])
            np.random.shuffle(idx)

            for epoch in range(self.epoch_size):
                for batch in range(n_batch):
                    i_start = batch * self.minibatch_size
                    i_end = (batch + 1) * self.minibatch_size
                    x_batch = x[idx[i_start:i_end]]

                    self.sess.run(minimizer, feed_dict={
                        input:x_batch, drop:self.est_dropout_ratio})

                if (epoch + 1) % 100 == 0:

                    loss_val = self.sess.run(loss, feed_dict={input:x, drop:0})
                    rec_loss = self.sess.run(self.comp_net.reconstruction_error(input, x_dash), feed_dict={input:x, drop:0})
                    print(" epoch {}/{} : loss = {:.3f}; rec_loss = {:.3f}".format(epoch + 1, self.epoch_size, loss_val, rec_loss))
                    
            
            GAMMA=self.sess.run(gamma, feed_dict={self.input:x[1:10,:]})

            fix = self.gmm.fix_operator()
            self.sess.run(fix, feed_dict={input:x, drop:0})
            self.energy = self.gmm.Calculate_Energy(z)

    def construct_Dict(self, x):
        """ Calculate Dict for samples in X.
        #######################################################
        Parameters
            x : (n_samples, n_features)
        #######################################################
        Returns
            Dict : (2*clusters_num, n_features)
        """
        if self.sess is None:
            raise Exception("Session does not exist!!!!")

        if self.normalize:
            x = self.scaler.transform(x)

        Gamma = self.sess.run(self.gamma, feed_dict={self.input:x})
        clusters_num = Gamma.shape[1]
        Dict_index = []
        cluster = np.argmax(Gamma,axis=1)
        for i in range(clusters_num):
            cluster_index = np.where(cluster==i)
            if len(cluster_index[0]) != 0:
                atom_1 = np.where(Gamma==Gamma[cluster_index,i].max())
                atom_2 = np.where(Gamma==Gamma[cluster_index,i].min())
                Dict_index.append(atom_1[0][0])
                Dict_index.append(atom_2[0][0])
        #Dict_index = np.array(Dict_index)
        S = self.sess.run(self.z, feed_dict={self.input:x})
        Dict = S[Dict_index,:]
        return Dict, S
    def construct_Dict_with_Ori_dim(self, x):
        """ Calculate Dict for samples in X.
        #######################################################
        Parameters
            x : (n_samples, n_features)
        #######################################################
        Returns
            Dict : (2*clusters_num, n_features)
        """
        if self.sess is None:
            raise Exception("Session does not exist!!!!")

        if self.normalize:
            x = self.scaler.transform(x)

        Gamma = self.sess.run(self.gamma, feed_dict={self.input:x})
        clusters_num = Gamma.shape[1]
        Dict_index = []
        cluster = np.argmax(Gamma,axis=1)
        for i in range(clusters_num):
            cluster_index = np.where(cluster==i)
            if len(cluster_index[0]) != 0:
                atom_1 = np.where(Gamma==Gamma[cluster_index,i].max())
                atom_2 = np.where(Gamma==Gamma[cluster_index,i].min())
                Dict_index.append(atom_1[0][0])
                Dict_index.append(atom_2[0][0])
        S = x
        Dict = S[Dict_index,:]
        return Dict, S