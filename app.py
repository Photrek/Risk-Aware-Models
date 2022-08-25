import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image


from model import VAEModel

def load_model():
    
    # Initialise model
    model = VAEModel(z_dim=256, n_filter_base=64, seed=1, dtype='float64')
    
    # Forward pass
    x = tf.random.normal((128,32,32,3))    
    y = model(x, input_type='cifar10')
    
    # Load weights
    model.load_weights('checkpoints/cp.ckpt')
    
    return model

# show = st.image(image, use_column_width=True)
st.title("Coupled Variational AutoEncoder demo")

model = load_model()

image = Image.open('demo_image.png')
st.image(image, 'Demo Image', use_column_width=True)
# We preprocess the image to fit in algorithm.
# image = np.asarray(image)
image_tensor = tf.expand_dims(tf.keras.preprocessing.image.img_to_array(image)[:, :, :3], axis=0)

gen_images = []
num_images = 9
dim = int(num_images ** 0.5)
for i in range(1, 10):
    recons_logits, z_sample, mean, logvar = model(image_tensor/255, input_type='cifar10')
    if i == int(num_images/2) + 1: # middle images
        gen_images.append(tf.math.sigmoid(model.decoder(mean)))
    else:
        gen_images.append(tf.math.sigmoid(recons_logits))
    
fig, axs = plt.subplots(dim, dim)

for i, (ax, im) in enumerate(zip(axs.reshape(-1), gen_images)):
    ax.imshow(im[0].numpy())
    ax.axis('off')
    
st.pyplot(fig)

