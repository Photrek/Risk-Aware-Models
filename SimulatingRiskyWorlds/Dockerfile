FROM nvidia/cuda:12.0.1-base-ubuntu22.04

RUN apt-get update && \
  apt install sudo && \
  apt-get install -y --no-install-recommends python3.10 python3-pip && \
  apt-get install -y ffmpeg libsm6 libxext6 
#RUN apt-get install -y 

# Setup Docker user
RUN adduser --gecos "" --disabled-password docker-user && adduser docker-user sudo
COPY entrypoint.sh /root/
ENTRYPOINT /root/entrypoint.sh
RUN echo 'docker-user:Docker!' | chpasswd

RUN python3.10 -m pip install torch torchvision torchaudio torchmetrics \
  cma tqdm matplotlib numpy opencv-python jupyterlab jupyterlab-vim ipywidgets \ 
  plotly scipy lightning[extra] torchmetrics pandas scikit-learn

RUN echo 'export PS1="\[$(tput bold)\]\[\033[38;5;2m\]PHOTREK-Gym-Docker \[$(tput sgr0)\]:\W\\$ \[$(tput sgr0)\]"' >> ~/.bashrc
RUN echo 'set -o vi' >> ~/.bashrc

# lightning[extra]: the "extra" enables tensorboard

CMD ["bash"] 

