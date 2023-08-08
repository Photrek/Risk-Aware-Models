FROM nvidia/cuda:12.0.1-base-ubuntu22.04

RUN apt-get update && \
  apt-get install -y --no-install-recommends python3.10 python3-pip && \
  apt-get install -y ffmpeg libsm6 libxext6 
#RUN apt-get install -y 

RUN python3.10 -m pip install torch torchvision torchaudio torchmetrics \
  cma tqdm matplotlib numpy opencv-python jupyterlab jupyterlab-vim ipywidgets \ 
  plotly scipy lightning[extra] torchmetrics pandas scikit-learn

# lightning[extra]: the "extra" enables tensorboard

CMD ["bash"] 

