# 7002

import base64
import io

import tensorflow as tf
import numpy as np
from VAEModelMNIST import VAEModelMNIST
import matplotlib.pyplot as plt

import data_generator_pb2 as pb2
import data_generator_pb2_grpc as pb2_grpc
import grpc
from concurrent import futures
import time
import os

class ServiceDefinition(pb2_grpc.ServiceDefinitionServicer):

    def __init__(self):
        self.s = ''
        self.response = None
    
    def GenerateImage(self, request, context):
        # Getting class from client message
        chosen_class = int(request.k)

        # Load model
        checkpoint_path = 'checkpoints/cp.ckpt'
        model = VAEModelMNIST(2, 64, 123456, 'float64')
        model.load_weights(checkpoint_path)

        print('Weights loaded')

        # Import data
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

        sample_index = np.random.choice(np.squeeze(np.argwhere(y_train == chosen_class)))

        x_train_tf = tf.expand_dims(tf.convert_to_tensor(x_train), axis=-1)
        x_train_tf = x_train_tf/255

        img_logits, z_sample, mu, sigma = model(tf.expand_dims(x_train_tf[sample_index, :, :, :], axis=0))

        plt.imshow(tf.math.sigmoid(img_logits)[0], cmap='Greys')
        plt.axis('off')
        plt.show()
        # plt.savefig('generated_img.png')
        my_stringIObytes = io.BytesIO()
        plt.savefig(my_stringIObytes, bbox_inches='tight', format='png')
        my_stringIObytes.seek(0)
        my_base64pngData = base64.b64encode(my_stringIObytes.read())

        print('Image generated')

        self.response = pb2.StringResponse()
        self.response.s = my_base64pngData

        return self.response
            
    def DownloadFile(self, request, context):
        chunk_size=1024

        filepath = f'{request.filename}{request.extension}'
        if os.path.exists(filepath):
            with open(filepath, mode='rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if chunk:
                        entry_response = pb2.FileResponse(chunk_data=chunk)
                        yield entry_response
                    else:
                        return


def main():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_ServiceDefinitionServicer_to_server(ServiceDefinition(), server)
    server.add_insecure_port('[::]:7002')
    server.start()
    print("Server listening on 0.0.0.0:{}".format(7002))
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    main()