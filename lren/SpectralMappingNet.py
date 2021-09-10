import tensorflow as tf

class SpectralMappingNet:

    def __init__(self, hidden_layer_sizes, activation=tf.nn.tanh):

        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation

    def Encoder(self, x):
        self.input_size = x.shape[1]

        with tf.variable_scope("Encoder"):
            z = x
            n_layer = 0
            for size in self.hidden_layer_sizes[:-1]:
                n_layer += 1
                z = tf.layers.dense(z, size, activation=self.activation,
                    name="layer_{}".format(n_layer))


            n_layer += 1
            z = tf.layers.dense(z, self.hidden_layer_sizes[-1],
                name="layer_{}".format(n_layer))

        return z

    def Decoder(self, z):
        with tf.variable_scope("Decoder"):
            n_layer = 0
            for size in self.hidden_layer_sizes[:-1][::-1]:
                n_layer += 1
                z = tf.layers.dense(z, size, activation=self.activation,
                    name="layer_{}".format(n_layer))


            n_layer += 1
            x_dash = tf.layers.dense(z, self.input_size,
                name="layer_{}".format(n_layer))

        return x_dash

    def loss(self, x, x_dash):
        def euclid_norm(x):
            return tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))

        # Calculate Euclid norm, distance
        norm_x = euclid_norm(x)
        norm_x_dash = euclid_norm(x_dash)
        dist_x = euclid_norm(x - x_dash)
        dot_x = tf.reduce_sum(x * x_dash, axis=1)

        min_val = 1e-3
        loss_E = dist_x  / (norm_x + min_val)
        loss_C = 0.5 * (1.0 - dot_x / (norm_x * norm_x_dash + min_val))

        loss_OPD = self.OPD(x, x_dash)
        return tf.concat([loss_E[:,None], loss_OPD[:,None]],axis=1)

    def extract_feature(self, x, x_dash, z_c):
        z_r = self.loss(x, x_dash)
        return tf.concat([z_c], axis=1)

    def inference(self, x):

        with tf.variable_scope("CompNet"):
            z_c = self.Encoder(x)
            x_dash = self.Decoder(z_c)

            z = self.extract_feature(x, x_dash, z_c)

        return z, x_dash

    def reconstruction_error(self, x, x_dash):
        return tf.reduce_mean(tf.reduce_sum(
            tf.square(x - x_dash), axis=1), axis=0)# + self.OPD(x,x_dash)
    def OPD(self, ri, rj):
        """
        ri   # pixels*bands
        rj   # pixels*bands
        """
        with tf.variable_scope("OPD"):
            ri = tf.transpose(ri)  # 205*
            rj = tf.transpose(rj)  # 205*
            L = ri.get_shape()[0]  # 205
            I = tf.ones([L, 1], tf.float32)
            ones = tf.ones([L, 1], tf.float32)
            diag1 = tf.diag_part(tf.matmul(tf.transpose(ri), ri))
            diag2 = tf.diag_part(tf.matmul(tf.transpose(rj), rj))
            diag1_inv = tf.div(1., 1e-6+diag1)
            diag2_inv = tf.div(1., 1e-6+diag2)

            
            diag1_inv = diag1_inv[:, None]
            diag2_inv = diag2_inv[:, None]
            Pri_perp = I - ri * ri * tf.matmul(ones, tf.transpose(diag1_inv))
            Prj_perp = I - rj * rj * tf.matmul(ones, tf.transpose(diag2_inv))
    
            val = tf.reduce_sum(ri * Prj_perp * ri, axis=0) + \
                tf.reduce_sum(rj * Pri_perp * rj, axis=0)
    
            val = tf.sqrt(val)
    
            return val