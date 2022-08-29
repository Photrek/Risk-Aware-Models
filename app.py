import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image


from model import VAEModel

def load_model(path):
    
    # Initialise model
    model = VAEModel(z_dim=256, n_filter_base=64, seed=1, dtype='float64')
    
    # Forward pass
    x = tf.random.normal((128,32,32,3))    
    y = model(x, input_type='cifar10')
    
    # Load weights
    # model.load_weights('checkpoints/cp.ckpt')
    model.load_weights(path)
    
    return model

def sample(mean, logvar):
    # Reparameterize
    eps = tf.cast(tf.random.normal(shape=mean.shape), 'float64')
    z_sample = eps * tf.exp(logvar * .5) + mean
    return z_sample

# show = st.image(image, use_column_width=True)
st.title("Coupled Variational AutoEncoder demo")

coupling = st.sidebar.selectbox('Coupling level', (0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6))

if not coupling:
    coupling =0.1
path = f'checkpoints/{coupling}/cp.ckpt'
model = load_model(path)

image = Image.open('demo_image.png')

# We preprocess the image to fit in algorithm.
image_tensor = tf.expand_dims(tf.keras.preprocessing.image.img_to_array(image)[:, :, :3], axis=0)
_, mean, logvar = model.encoder(image_tensor/255)


# Generating reconstructions
gen_images = []
num_images = 9
dim = int(num_images ** 0.5)
for i in range(1, 10):
    z_sample = sample(mean, logvar)
    recons_logits = model.decoder(z_sample)
    if i == int(num_images/2) + 1: # middle images
        gen_images.append(tf.math.sigmoid(model.decoder(mean)))
    else:
        gen_images.append(tf.math.sigmoid(recons_logits))
    
# Original vs reconstructions    
fig, axs = plt.subplots(dim, dim)
for i, (ax, im) in enumerate(zip(axs.reshape(-1), gen_images)):    
    if i == int(num_images/2):
        ax.imshow(image)
    else:
        ax.imshow(im[0].numpy())
    ax.axis('off')

st.subheader('Original vs reconstructions')    
st.pyplot(fig)

# Average reconstruction vs reconstructions
fig, axs = plt.subplots(dim, dim)
for i, (ax, im) in enumerate(zip(axs.reshape(-1), gen_images)):    
    ax.imshow(im[0].numpy())
    ax.axis('off')

st.subheader('Average reconstruction vs reconstructions')    
st.pyplot(fig)
