import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image


from model import VAEModel

def load_model():
    
    # Initialise model
    model = VAEModel(z_dim=160, n_filter_base=64, seed=1, dtype='float64')
    
    # Forward pass
    x = tf.random.normal((128,32,32,3))    
    y = model(x, input_type='cifar10')
    
    # Load weights
    model.load_weights('checkpoints/cp.ckpt')
    
    return model

# show = st.image(image, use_column_width=True)
st.title("CVAE")

model = load_model()

## Image upload
st.sidebar.title("Upload Image")
#Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)
#Choose your own image
uploaded_file = st.sidebar.file_uploader(" ",type=['png', 'jpg', 'jpeg'] )

if uploaded_file is not None:
    
    u_img = Image.open(uploaded_file)
    st.image(u_img, 'Uploaded Image', use_column_width=True)


    image_tensor = tf.expand_dims(tf.keras.preprocessing.image.img_to_array(u_img)[:, :, :3], axis=0)
    
    gen_image = model(image_tensor/255, input_type='cifar10')
    
    st.image(tf.math.sigmoid(gen_image[0][0]).numpy(), 'Generated Image', use_column_width=True)

## Demo file
st.sidebar.title("Run demo")
if st.sidebar.button('Go!'):
    
    image = Image.open('demo_image.png')
    st.image(image, 'Demo Image', use_column_width=True)
    # We preprocess the image to fit in algorithm.
    # image = np.asarray(image)
    image_tensor = tf.expand_dims(tf.keras.preprocessing.image.img_to_array(image)[:, :, :3], axis=0)
    gen_image = model(image_tensor/255, input_type='cifar10')
    st.image(tf.math.sigmoid(gen_image[0][0]).numpy(), 'Generated Image', use_column_width=True)