# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

class GaussianMixtureModel:
    def __init__(self, n_comp):
        self.n_comp = n_comp
        self.Phi = self.Mu = self.Sigma = None
        self.training = False

    def create_variables(self, n_features):
        with tf.variable_scope("GMM"):
            Phi = tf.Variable(tf.zeros(shape=[self.n_comp]),
                dtype=tf.float32, name="Phi")
            Mu = tf.Variable(tf.zeros(shape=[self.n_comp, n_features]),
                dtype=tf.float32, name="Mu")
            Sigma = tf.Variable(tf.zeros(
                shape=[self.n_comp, n_features, n_features]),
                dtype=tf.float32, name="Sigma")
            L_Cholesky = tf.Variable(tf.zeros(
                shape=[self.n_comp, n_features, n_features]),
                dtype=tf.float32, name="L_Cholesky")

        return Phi, Mu, Sigma, L_Cholesky

    def Gaussian_Mixture_Model_Parameter_Estimation(self, z, gamma):
        with tf.variable_scope("Gaussian_Mixture_Model"):

            # Calculate Mu, Sigma with Einstein summation convention
            gamma_sum = tf.reduce_sum(gamma, axis=0)
            self.Phi = Phi = tf.reduce_mean(gamma, axis=0)
            self.Mu = Mu = tf.einsum('ik,il->kl', gamma, z) / gamma_sum[:,None]
            z_centered = tf.sqrt(gamma[:,:,None]) * (z[:,None,:] - Mu[None,:,:])
            self.Sigma = Sigma = tf.einsum(
                'ikl,ikm->klm', z_centered, z_centered) / gamma_sum[:,None,None]

            # Cholesky decomposition
            n_features = z.shape[1]
            min_vals = tf.diag(tf.ones(n_features, dtype=tf.float32)) * 1e-6
            self.L_Cholesky = tf.cholesky(Sigma + min_vals[None,:,:])

        self.training = False

    def fix_operator(self):
        Phi, Mu, Sigma, L_Cholesky = self.create_variables(self.Mu.shape[1])

        op = tf.group(
            tf.assign(Phi, self.Phi),
            tf.assign(Mu, self.Mu),
            tf.assign(Sigma, self.Sigma),
            tf.assign(L_Cholesky, self.L_Cholesky)
        )

        self.Phi, self.Phi_org = Phi, self.Phi
        self.Mu, self.Mu_org = Mu, self.Mu
        self.Sigma, self.Sigma_org = Sigma, self.Sigma
        self.L_Cholesky, self.L_Cholesky_org = L_Cholesky, self.L_Cholesky

        self.training = False

        return op

    def Calculate_Energy(self, z):
        if self.training and self.Phi is None:
            self.Phi, self.Mu, self.Sigma, self.L_Cholesky = self.create_variable(z.shape[1])

        with tf.variable_scope("Gaussian_Mixture_Model_Energy"):
            z_centered = z[:,None,:] - self.Mu[None,:,:]  #ikl
            v = tf.matrix_triangular_solve(self.L_Cholesky, tf.transpose(z_centered, [1, 2, 0]))  # kli
            log_det_Sigma = 2.0 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.L_Cholesky)), axis=1)
            d = z.get_shape().as_list()[1]
            logits = tf.log(self.Phi[:,None]) - 0.5 * (tf.reduce_sum(tf.square(v), axis=1)
                + d * tf.log(2.0 * np.pi) + log_det_Sigma[:,None])
            energies = - tf.reduce_logsumexp(logits, axis=0)

        return energies

    def Cov_Diag_Loss(self):
        with tf.variable_scope("Gaussian_Mixture_Model_Diag_Loss"):
            diag_loss = tf.reduce_sum(tf.divide(1, tf.matrix_diag_part(self.Sigma)))

        return diag_loss
