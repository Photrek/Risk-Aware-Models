from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl
from tensorflow.keras.layers import Input, InputLayer, Lambda, Reshape, \
                                    Dropout, Flatten, Dense, Conv2D, \
                                    Conv2DTranspose

import tensorflow as tf

class Sampler_Z(tfkl.Layer):   

    def call(self, inputs):
        mean, logvar = inputs
        # Reparameterize
        eps = tf.random.normal(shape=mean.shape)
        z_sample = eps * tf.exp(logvar * .5) + mean
        #tf.print(tf.math.reduce_mean(eps))
        return z_sample

class EncoderZ_5(tfkl.Layer):
    # n_filter_base is a bogus argument for now
    def __init__(self, z_dim, n_filter_base, dtype, name='decoder', **kwargs):
        super(EncoderZ_5, self).__init__(name=name, **kwargs)

        self.conv_layers = tf.keras.Sequential(
          [
              tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
              tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu', dtype=dtype),
              tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu', dtype=dtype),
              tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=(2, 2), activation='relu', dtype=dtype),
              tf.keras.layers.Flatten()
          ]
        )
        
        self.dense_mean = Dense(
              z_dim, 
              activation=None, 
              name='z_mean', 
              dtype=dtype
              )
        
        self.dense_raw_stddev = Dense(
            z_dim, 
            activation=None,
            name='z_raw_stddev', 
            dtype=dtype
            )
        
        self.sampler_z = Sampler_Z()
      
    def call(self, x_input):
      z = self.conv_layers(x_input)
      mean = self.dense_mean(z)
      logvar = self.dense_raw_stddev(z)
      z_sample = self.sampler_z((mean, logvar))
      return z_sample, mean, logvar

class DecoderX_5(tfkl.Layer):

    def __init__(self, z_dim, n_filter_base, dtype, name='decoder', **kwargs):
        super(DecoderX_5, self).__init__(name=name, **kwargs)

        self.deconv_layers = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(z_dim,)),
                tf.keras.layers.Dense(units=4*4*128, activation=tf.nn.relu, dtype=dtype),
                tf.keras.layers.Reshape(target_shape=(4, 4, 128)),
                tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation='relu', dtype=dtype),
                tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu', dtype=dtype),
                tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu', dtype=dtype),
                tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same', dtype=dtype),
            ]
        )
    

    def call(self, z):
      return self.deconv_layers(z)

class VAEModel(tfk.Model):
    '''Convolutional variational autoencoder base model.

      input_type: str
    The type of inputs to the neural network which affect the architecture e.g. mnist or cifar10.
    '''
    
    def __init__(self, z_dim, n_filter_base, seed, dtype, input_type='mnist'):
        super(VAEModel, self).__init__()
        self.encoder = EncoderZ_5(z_dim, n_filter_base, dtype)
        self.decoder = DecoderX_5(z_dim, n_filter_base, dtype)
        return
    
    #@tf.function
    def sample(self, z_sample, input_type):
        x_recons_logits = self.decoder(z_sample)
        sample_images = tf.sigmoid(x_recons_logits)  # predictions
        return sample_images
    
    def call(self, x_input, input_type):
        z_sample, mean, logvar = self.encoder(x_input)
        x_recons_logits = self.decoder(z_sample)
        # tf.print(x_recons_logits.shape)
        return x_recons_logits, z_sample, mean, logvar