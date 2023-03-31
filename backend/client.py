# tutorial
# 7003

import os
import io
import sys
import grpc
import base64
from PIL import Image, ImageDraw

sys.path.append("./service_spec")
import data_generator_pb2 as pb2
import data_generator_pb2_grpc as pb2_grpc

#sudo cp /home/hxyue1/{checkpoint,cp.ckpt.data-00000-of-00001,cp.ckpt.index} /home/blake_anderton/ram/checkpoints

def run():
    with grpc.insecure_channel('34.136.24.206:7002') as channel:
    # with grpc.insecure_channel('localhost:7002') as channel:

        # Taking input arguments
        dataset_type = sys.argv[1]
        chosen_class = sys.argv[2]

        # Initiating connection with GRPC service
        stub = pb2_grpc.ServiceDefinitionStub(channel)

        # Passing to server method and getting response
        response = stub.GenerateImage(pb2.Input(d=dataset_type, k=chosen_class))
        
        # Convert base64 encoded response string back to png
        response_string = response.s

        img = Image.open(io.BytesIO(base64.decodebytes(bytes(response_string, "utf-8"))))
        img.save('generated_img.png')

    
if __name__ == '__main__':
    run()
