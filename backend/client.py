# tutorial
# 7003

import os
import sys
import grpc

sys.path.append("./service_spec")
import data_generator_pb2 as pb2
import data_generator_pb2_grpc as pb2_grpc

#sudo cp /home/hxyue1/{checkpoint,cp.ckpt.data-00000-of-00001,cp.ckpt.index} /home/blake_anderton/ram/checkpoints

def run():
    # with grpc.insecure_channel('104.154.63.152:7002') as channel:
    with grpc.insecure_channel('34.136.24.206:7002') as channel:
        stub = pb2_grpc.ServiceDefinitionStub(channel)

        chosen_class = sys.argv[1]
        response = stub.GenerateImage(pb2.Class(k=chosen_class))
        print(response)

        for entry_response in stub.DownloadFile(pb2.MetaData(filename='generated_img', extension='.png')):
            with open('generated_img.png', mode="ab") as f:
                f.write(entry_response.chunk_data)

    
if __name__ == '__main__':
    run()
